import re
import os
import json
import time
import requests
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
import textstat
import yake
import joblib
import warnings
from typing import List, Tuple, Dict, Optional
warnings.filterwarnings('ignore')


MODEL_PATH = os.path.join(os.getcwd(), "models")

try:
    label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.joblib"))
    tfidf_vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidf_vectorizer.joblib"))
    voting_classifier = joblib.load(os.path.join(MODEL_PATH, "voting_classifier.joblib"))
    print("✅ Loaded pretrained classification models successfully!\n")
except Exception as e:
    print(f"⚠️ Classification models not found: {e}")
    print("   Continuing without ML classification feature\n")
    label_encoder, tfidf_vectorizer, voting_classifier = None, None, None


def classify_research_paper(title: str, abstract: str):
    """
    Predicts the category of a research paper using pre-trained TF-IDF + Voting Classifier.
    """
    if not all([label_encoder, tfidf_vectorizer, voting_classifier]):
        return "Unknown"

    try:
        text_input = title + " " + abstract
        X_vec = tfidf_vectorizer.transform([text_input])
        y_pred = voting_classifier.predict(X_vec)
        label = label_encoder.inverse_transform(y_pred)[0]
        return label
    except Exception as e:
        print(f"⚠️ Classification failed: {e}")
        return "Unknown"

####

class SemanticScholarSearch:
    def __init__(self):
        pass

    def payload(self, keyword, page=1, min_year=2018, max_year=2025):
        """Create search request payload"""
        headers = {
            "Connection": "keep-alive",
            "sec-ch-ua": '"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
            "Cache-Control": "no-cache,no-store,must-revalidate,max-age=-1",
            "Content-Type": "application/json",
            "sec-ch-ua-mobile": "?1",
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Mobile Safari/537.36",
            "X-S2-UI-Version": "20166f1745c44b856b4f85865c96d8406e69e24f",
            "sec-ch-ua-platform": '"Android"',
            "Accept": "*/*",
            "Origin": "https://www.semanticscholar.org",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "Referer": "https://www.semanticscholar.org/search",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        }

        data = json.dumps({
            "queryString": f"{keyword.lower()}",
            "page": page,
            "pageSize": 10,
            "sort": "relevance",
            "authors": [],
            "coAuthors": [],
            "venues": [],
            "yearFilter": {"min": min_year, "max": max_year},
            "requireViewablePdf": False,
            "fieldsOfStudy": [],
            "hydrateWithDdb": True,
            "includeTldrs": True,
            "performTitleMatch": True,
            "includeBadges": True,
            "getQuerySuggestions": False,
        })

        try:
            response = requests.post(
                "https://www.semanticscholar.org/api/1/search",
                headers=headers,
                data=data,
                timeout=10
            )
            return response
        except Exception as e:
            return None

    def parse_results(self, response):
        """Parse search results"""
        final_result = []
        
        try:
            output = response.json().get("results", [])
            
            for paper in output:
                result = {"title": "", "link": "", "year": ""}
                # title
                try:
                    result["title"] = paper["title"]["text"]
                except Exception:
                    result["title"] = paper.get("title", "") or ""

                # year
                if "year" in paper and paper["year"]:
                    try:
                        result["year"] = paper["year"]["text"]
                    except Exception:
                        result["year"] = str(paper.get("year", "N/A"))
                else:
                    result["year"] = "N/A"

                # link
                if "primaryPaperLink" in paper and paper["primaryPaperLink"]:
                    result["link"] = paper["primaryPaperLink"].get("url", "no_link_found")
                elif paper.get("alternatePaperLinks"):
                    alt = paper.get("alternatePaperLinks")
                    if isinstance(alt, list) and alt:
                        result["link"] = alt[0].get("url", "no_link_found")
                    else:
                        result["link"] = "no_link_found"
                else:
                    result["link"] = "no_link_found"

                final_result.append(result)
                
            df = pd.DataFrame(final_result)
            return df
        except Exception as e:
            return pd.DataFrame()
   
    def search_papers(self, keyword, max_pages=2, min_year=2018, max_year=2025, api_wait=3):
        """Search Semantic Scholar for papers"""
        all_pages = []
        for page in range(1, max_pages + 1):
            response = self.payload(keyword, page=page, min_year=min_year, max_year=max_year)
            
            if response and getattr(response, "status_code", None) == 200:
                result_df = self.parse_results(response)
                if not result_df.empty:
                    all_pages.append(result_df)
                time.sleep(api_wait)

        if all_pages:
            df = pd.concat(all_pages, ignore_index=True)
            return df
        else:
            return pd.DataFrame()


###

class TopicExtractor:
    """Extract main topics from research paper with domain priority"""
    
    def __init__(self):
        try:
            self.rake = Rake()
            self.yake_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.8, top=10)
        except Exception:
            self.rake = None
            self.yake_extractor = None
    
    def extract_title(self, text):
        """Extract paper title"""
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 20 and len(line) < 200 and not line[0].isdigit():
                line = re.sub(r'[^\w\s-]', ' ', line)
                line = re.sub(r'\s+', ' ', line)
                return line.strip()
        return ""
    
    def extract_abstract(self, text):
        """Extract abstract section"""
        abstract_match = re.search(r'(?i)abstract[:\s]+(.*?)(?=\n\n[A-Z0-9]|introduction|1\.)', text, re.DOTALL)
        if abstract_match:
            return abstract_match.group(1).strip()[:1500]
        return ""
    
    def is_methodology_term(self, term):
        """Check if term is a generic ML/DL methodology term"""
        term_lower = term.lower()
        
        methodology_terms = [
            'machine learning', 'deep learning', 'artificial intelligence', 'neural network',
            'convolutional neural network', 'recurrent neural network', 'algorithm',
            'classification', 'regression', 'supervised learning', 'unsupervised learning',
            'reinforcement learning', 'feature extraction', 'data mining', 'predictive model',
            'cnn', 'rnn', 'lstm', 'transformer', 'random forest', 'svm', 'decision tree',
            'state of the art', 'novel approach', 'comparative study', 'performance evaluation',
            'transfer learning', 'ensemble method', 'gradient boosting', 'support vector',
            'k nearest neighbor', 'naive bayes', 'logistic regression', 'linear regression'
        ]
        
        return any(method in term_lower for method in methodology_terms)
    
    def extract_domain_terms(self, text):
        """Extract domain-specific application terms"""
        text_lower = text[:3000].lower()
        
        domain_patterns = {
            'cancer|tumor|oncology|malignant|breast cancer|lung cancer': 'Cancer detection',
            'diabetes|glucose|insulin|blood sugar': 'Diabetes prediction',
            'covid|coronavirus|sars-cov-2|pandemic': 'COVID-19 detection',
            'mpox|monkeypox': 'Mpox detection',
            'pneumonia|lung disease|respiratory': 'Pneumonia detection',
            'alzheimer|dementia|cognitive': 'Alzheimer detection',
            'heart disease|cardiovascular|cardiac': 'Heart disease prediction',
            'skin lesion|dermatology|melanoma': 'Skin disease detection',
            'credit card fraud': 'Credit card fraud detection',
            'fraud detection|fraudulent transaction|anomaly detection': 'Fraud detection',
            'financial fraud|bank fraud|payment fraud': 'Financial fraud detection',
            'stock price|stock market prediction|stock forecast': 'Stock market prediction',
            'intrusion detection|network security|network attack': 'Intrusion detection',
            'malware|virus detection|malicious software': 'Malware detection',
            'spam detection|email filtering|spam classification': 'Spam detection',
            'sentiment analysis|opinion mining|emotion detection': 'Sentiment analysis',
            'text classification|document classification': 'Text classification',
            'face recognition|facial recognition': 'Face recognition',
            'object detection|object recognition': 'Object detection',
            'image classification|image recognition': 'Image classification',
            'recommendation system|recommender|collaborative filtering': 'Recommendation system',
        }
        
        detected_domains = []
        for pattern, domain in domain_patterns.items():
            if re.search(pattern, text_lower):
                detected_domains.append(domain)
        
        return detected_domains
    
    def extract_main_topic(self, text):
        """Extract main topic with domain priority over methodology"""
        title = self.extract_title(text)
        abstract = self.extract_abstract(text)
        
        domain_terms = self.extract_domain_terms(text)
        if domain_terms:
            return domain_terms[0], domain_terms[:3]
        
        title_topics = []
        if title:
            title_clean = title.lower()
            
            for method_term in ['using', 'with', 'based on', 'via', 'through', 'by', 
                              'machine learning', 'deep learning', 'neural network', 
                              'cnn', 'rnn', 'lstm', 'algorithm', 'algorithms',
                              'state of the art', 'novel', 'approach']:
                title_clean = re.sub(rf'\b{re.escape(method_term)}\b', '', title_clean, flags=re.IGNORECASE)
            
            title_clean = re.sub(r'\s+', ' ', title_clean).strip()
            
            parts = re.split(r'[:,\-]', title_clean)
            for part in parts:
                part = part.strip()
                if len(part) > 10 and not self.is_methodology_term(part):
                    title_topics.append(part)
        
        analysis_text = f"{title} {abstract}"
        if len(analysis_text) < 100:
            analysis_text = text[:2000]
        
        extracted_topics = []
        
        if self.rake:
            try:
                self.rake.extract_keywords_from_text(analysis_text)
                rake_phrases = self.rake.get_ranked_phrases()[:10]
                for phrase in rake_phrases:
                    if not self.is_methodology_term(phrase) and len(phrase) > 8:
                        extracted_topics.append(phrase)
            except Exception:
                pass
        
        all_topics = title_topics + extracted_topics
        
        unique_topics = []
        seen = set()
        for topic in all_topics:
            topic_clean = topic.lower().strip()
            if topic_clean and len(topic_clean) > 8 and topic_clean not in seen:
                seen.add(topic_clean)
                unique_topics.append(topic)
        
        if unique_topics:
            return unique_topics[0], unique_topics[:3]
        else:
            if title:
                main_topic = title.lower()
                main_topic = re.sub(r'\b(a|an|the|using|with|for|on)\b', '', main_topic)
                main_topic = re.sub(r'\s+', ' ', main_topic).strip()
                return main_topic, [main_topic]
            else:
                return "research paper", ["research paper"]

#####

class ScientificTermExtractor:
    def __init__(self):
        self.generic_terms = self._load_generic_terms()

    def _load_generic_terms(self):
        generic_terms = set()
        blacklist_file = "C:/Users/karti/Desktop/CAPSTONE/10000 generic english words.txt"

        try:
            if os.path.exists(blacklist_file):
                with open(blacklist_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            generic_terms.add(word)
                print(f"✅ Loaded {len(generic_terms)} generic terms from blacklist")
            else:
                print(f"❌ Blacklist file not found: {blacklist_file}")
        except Exception as e:
            print(f"❌ Error loading blacklist: {e}")

        return generic_terms

    def extract_scientific_terms(self, text: str) -> List[str]:
        all_terms = set()
        pattern_terms = self._extract_multiword_patterns(text)
        all_terms.update(pattern_terms)
        tfidf_terms = self._extract_with_tfidf(text)
        all_terms.update(tfidf_terms)
        suffix_terms = self._extract_with_suffix_patterns(text)
        all_terms.update(suffix_terms)
        acronym_terms = self._extract_acronyms(text)
        all_terms.update(acronym_terms)
        filtered_terms = self._remove_duplicates(all_terms)
        final_terms = self._apply_blacklist_filter(filtered_terms)
        print(f"🔬 After blacklist filtering: {len(final_terms)} scientific terms")
        return sorted(list(final_terms))

    def _extract_multiword_patterns(self, text: str) -> List[str]:
        scientific_terms = set()
        try:
            cleaned_text = self._clean_text(text)
            words = cleaned_text.split()
            for i in range(len(words) - 1):
                word1, word2 = words[i], words[i + 1]
                if (
                    len(word1) > 4
                    and len(word2) > 4
                    and self._looks_technical(word1)
                    and self._looks_technical(word2)
                ):
                    term = f"{word1} {word2}"
                    if self._is_valid_term(term):
                        scientific_terms.add(term)
        except Exception as e:
            print(f"Pattern extraction failed: {e}")
        return list(scientific_terms)

    def _extract_with_tfidf(self, text: str, top_n: int = 15) -> List[str]:
        scientific_terms = []
        try:
            cleaned_text = self._clean_text(text)
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                max_features=20,
                min_df=2,
                max_df=0.6,
            )
            tfidf_matrix = vectorizer.fit_transform([cleaned_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            for term, score in zip(feature_names, scores):
                if score > 0.2 and self._looks_technical(term) and self._is_valid_term(term):
                    scientific_terms.append(term.lower())
        except Exception:
            pass
        return scientific_terms[:top_n]

    def _extract_with_suffix_patterns(self, text: str) -> List[str]:
        scientific_terms = set()
        try:
            technical_suffixes = {"ology", "ography", "ometry", "ics", "ation", "ment", "ness", "ity"}
            words = re.findall(r"\b[a-z]+\b", text.lower())
            for word in words:
                for suffix in technical_suffixes:
                    if word.endswith(suffix) and len(word) > 6:
                        scientific_terms.add(word)
        except Exception as e:
            print(f"Suffix extraction failed: {e}")
        return list(scientific_terms)

    def _extract_acronyms(self, text: str) -> List[str]:
        scientific_terms = set()
        try:
            acronym_pattern = r"\(([A-Z]{2,})\)"
            matches = re.findall(acronym_pattern, text)
            common_acronyms = {"PDF", "URL", "HTTP", "HTML", "CSS", "USA", "UK"}
            for acronym in matches:
                if acronym not in common_acronyms:
                    scientific_terms.add(acronym)
        except Exception as e:
            print(f"Acronym extraction failed: {e}")
        return list(scientific_terms)

    def _looks_technical(self, word_or_term: str) -> bool:
        term_lower = word_or_term.lower()
        if len(term_lower) < 5:
            return False
        technical_suffixes = {
            "ology", "ography", "ometry", "ics", "ation", "ment", "ness", "ity", "ical", "istic"
        }
        words = term_lower.split()
        for word in words:
            for suffix in technical_suffixes:
                if word.endswith(suffix):
                    return True
        technical_prefixes = {
            "neuro", "bio", "chem", "phys", "quantum", "nano", "multi", "hyper", "pseudo"
        }
        for word in words:
            for prefix in technical_prefixes:
                if word.startswith(prefix):
                    return True
        return False

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\S+@\S+", "", text)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            if (
                len(re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", line)) > 1
                or "@" in line
            ):
                continue
            cleaned_lines.append(line)
        cleaned_text = " ".join(cleaned_lines)
        cleaned_text = cleaned_text.lower()
        cleaned_text = re.sub(r"[^\w\s-]", " ", cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text.strip()

    def _is_valid_term(self, term: str) -> bool:
        term_lower = term.lower()
        if len(term_lower) < 5:
            return False
        if re.search(r"[0-9@]", term_lower):
            return False
        if any(bad in term_lower for bad in ["email", "gmail", "university", "college", "author"]):
            return False
        return True

    def _remove_duplicates(self, terms):
        unique_terms = set()
        term_list = sorted(list(terms), key=len, reverse=True)
        for term in term_list:
            term_lower = term.lower()
            is_duplicate = False
            for existing in unique_terms:
                existing_lower = existing.lower()
                if (
                    term_lower in existing_lower
                    or existing_lower in term_lower
                ) and abs(len(term_lower) - len(existing_lower)) < 3:
                    is_duplicate = True
                    break
                if term_lower == existing_lower:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_terms.add(term)
        return unique_terms

    def _apply_blacklist_filter(self, terms):
        if not self.generic_terms:
            print("⚠ No blacklist loaded - returning all terms")
            return terms
        scientific_terms = set()
        removed_count = 0
        for term in terms:
            term_lower = term.lower()
            words = term_lower.split()
            if len(words) > 1:
                generic_word_count = sum(1 for word in words if word in self.generic_terms)
                if generic_word_count / len(words) <= 0.5:
                    scientific_terms.add(term)
                else:
                    removed_count += 1
            else:
                if term_lower not in self.generic_terms:
                    scientific_terms.add(term)
                else:
                    removed_count += 1
        print(f"🗑 Removed {removed_count} generic terms using blacklist")
        return scientific_terms

####

class ResearchPaperSummarizer:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
        except Exception as e:
            print(f"⚠️ Transformer model load failed: {e}")
            self.tokenizer = None
            self.model = None
            self.summarizer = None
        
        try:
            self.rake = Rake()
        except Exception:
            self.rake = None
        
        try:
            self.yake_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.8, top=15)
        except Exception:
            self.yake_extractor = None
        
        try:
            self.scientific_extractor = ScientificTermExtractor()
        except Exception:
            self.scientific_extractor = None
        
        try:
            self.topic_extractor = TopicExtractor()
        except Exception:
            self.topic_extractor = None
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract all text from PDF file"""
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return None
    
    def clean_text(self, text):
        """Clean text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def extract_key_terms(self, text: str, top_n: int = 15) -> List[Tuple[str, float]]:
        print("🔑 Extracting key terms (context-aware)...")
        try:
            if not text:
                return []
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if not sentences:
                return []
            if self.scientific_extractor:
                scientific_terms = set(self.scientific_extractor.extract_scientific_terms(text))
            else:
                scientific_terms = set()
            scored_sentences = []
            for sent in sentences:
                sent_lower = sent.lower()
                matched_terms = [term for term in scientific_terms if term.lower() in sent_lower]
                if matched_terms:
                    score = len(matched_terms) * 2.0
                    try:
                        length_penalty = 1.0 - abs(len(sent.split()) - 20) / 40.0
                    except Exception:
                        length_penalty = 0.5
                    score *= max(0.5, length_penalty)
                    scored_sentences.append((sent.strip(), score))
            if not scored_sentences:
                print("⚠ No scientific-term-rich sentences found, using YAKE fallback...")
                all_terms = {}
                if self.yake_extractor:
                    try:
                        yake_terms = self.yake_extractor.extract_keywords(text)[:top_n]
                        for term, score in yake_terms:
                            all_terms[term] = max(0.1, min(1.0, 1 - score))
                    except Exception:
                        pass
                return sorted(all_terms.items(), key=lambda x: x[1], reverse=True)[:top_n]
            max_score = max(score for _, score in scored_sentences)
            normalized = [(sent, round(score / max_score, 3)) for sent, score in scored_sentences]
            normalized.sort(key=lambda x: x[1], reverse=True)
            top_sentences = normalized[:top_n]
            print(f"✅ Returning {len(top_sentences)} enriched key terms")
            return top_sentences
        except Exception as e:
            print(f"❌ Enhanced key term extraction failed: {e}")
            return []

    def identify_scientific_terms(self, text: str):
        """Identify scientific terms"""
        if not self.scientific_extractor:
            return []
        return self.scientific_extractor.extract_scientific_terms(text)

    def calculate_readability_score(self, text: str) -> float:
        """Compute readability score safely"""
        try:
            sample_text = text[:3000] if len(text) > 3000 else text
            flesch = textstat.flesch_reading_ease(sample_text)
            score = max(0, min(100, flesch))
            return score
        except Exception:
            return 50.0

    def calculate_structure_score(self, text: str) -> float:
        try:
            sections = {
                "abstract": 25,
                "introduction": 20,
                "method": 15,
                "methodology": 15,
                "results": 20,
                "conclusion": 15,
            }
            text_lower = text.lower() if text else ""
            score = sum(points for section, points in sections.items() if section in text_lower)
            return min(100, score)
        except Exception:
            return 50.0

    def calculate_citation_score(self, text: str) -> float:
        try:
            text = text or ""
            bracket_citations = len(re.findall(r"\[\s*\d+\s*\]", text))
            author_year_citations = len(re.findall(r"\(\s*[A-Za-z][A-Za-z\.\- ]{0,40},?\s*\d{4}\s*\)", text))
            et_al_citations = len(re.findall(r"\(\s*[A-Za-z]+ et al\.", text, re.IGNORECASE))
            has_references = 1 if re.search(r"references\s*[\.:]", text, re.IGNORECASE) else 0
            has_bibliography = 1 if re.search(r"bibliography\s*[\.:]", text, re.IGNORECASE) else 0
            total_citations = bracket_citations + author_year_citations + et_al_citations
            score = 0
            if total_citations >= 20:
                score += 50
            elif total_citations >= 10:
                score += 35
            elif total_citations >= 5:
                score += 20
            elif total_citations >= 1:
                score += 10
            if has_references or has_bibliography:
                score += 30
            citation_types = sum([1 for x in [bracket_citations, author_year_citations, et_al_citations] if x > 0])
            score += min(20, citation_types * 10)
            print(f"   Citations: {score}/100 (Found {total_citations} citations)")
            return min(100, score)
        except Exception as e:
            print(f"   Citation calculation failed: {e}")
            return 50.0

    
    def calculate_quality_scores(self, text: str) -> dict:
        readability = self.calculate_readability_score(text)
        structure = self.calculate_structure_score(text)
        citation = self.calculate_citation_score(text)
        overall = (readability * 0.3) + (structure * 0.4) + (citation * 0.3)
        return {
            "overall": round(overall, 1),
            "readability": round(readability, 1),
            "structure": round(structure, 1),
            "citation": round(citation, 1),
        }
    
    def analyze_paper(self, filepath: str) -> Dict:
        print(f"\n🎯 Starting analysis: {os.path.basename(filepath)}")
        try:
            text = self.extract_text_from_pdf(filepath)
            if text is None:
                raise ValueError("Failed to extract text from PDF.")

            key_terms = self.extract_key_terms(text)
            scientific_terms = self.identify_scientific_terms(text)

            
            scores = self.calculate_quality_scores(text)

            results = {
                "key_terms": [{"term": term, "score": float(score)} for term, score in key_terms],
                "scientific_terms": scientific_terms,
                "scores": scores,
                "text_length": len(text),
                "word_count": len(text.split()),
                "status": "success",
            }
            print("✅ Analysis completed successfully!")
            return results
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {"error": str(e), "status": "error"}

    def extract_section_headings(self, text):
        """Extract section headings"""
        if not text:
            return []
        standard_sections = [
            "Abstract", "Introduction", "Related Work", "Literature Review",
            "Background", "Motivation", "Methodology", "Methods", "Materials and Methods",
            "Proposed Method", "Proposed Methodology", "System Design",
            "Dataset", "Experiments", "Results", "Discussion",
            "Conclusion", "Future Work"
        ]
        
        found_sections = []
        for section in standard_sections:
            pattern = rf"(?:^|\n)\s*(?:\d+\.?\s*)?{re.escape(section)}s?\s*(?:\n|:|$)"
            if re.search(pattern, text, re.IGNORECASE):
                if section not in found_sections:
                    found_sections.append(section)
        return found_sections

    def extract_ml_methods_and_formulas(self, text):
        """Extract ML methods"""
        if not text:
            return []
        methods = []
        
        keywords = [
            "Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest",
            "CNN", "RNN", "LSTM", "ResNet", "DenseNet", "VGG", "BERT", "GPT",
            "Accuracy", "Precision", "Recall", "F1 Score", "MSE", "RMSE", "MAE",
            "Adam", "SGD", "ReLU", "Sigmoid", "Softmax", "Dropout",
            "Cross Entropy", "Data Augmentation", "Transfer Learning"
        ]
        
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
                methods.append(kw)

        return list(dict.fromkeys(methods))

    def extract_sections(self, text):
        if not text:
            return {k: "" for k in ["abstract", "introduction", "methodology", "results", "conclusion"]}

        sections = {}
        patterns = {
            "abstract": r"(?i)abstract[:\s]+(.*?)(?=\n\n[A-Z0-9]|introduction|1\.)",
            "introduction": r"(?i)(introduction|1\.?\s*introduction)(.*?)(?=\n\n[0-9]|method|approach|related work)",
            "methodology": r"(?i)(method|methodology|approach|materials and methods)(.*?)(?=\n\n[0-9]|result|experiment)",
            "results": r"(?i)(results?|findings?|experimental results?)(.*?)(?=\n\n[0-9]|discussion|conclusion)",
            "conclusion": r"(?i)(conclusion|discussion)(.*?)(?=references|acknowledgment|$)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            sections[key] = self.clean_text(match.group(2 if key != "abstract" else 1).strip()[:3000]) if match else ""

        return sections

    def abstractive_summarize(self, text, max_len=200, min_len=50):
        if not text or len(text) < 100 or not self.summarizer:
            return ""
        try:
            summary = self.summarizer(text, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
            return self.clean_text(summary[0]["summary_text"])
        except Exception:
            return ""

    def generate_summary(self, sections):
        summary = []
        for key, label in [("abstract","ABSTRACT"),("introduction","INTRODUCTION"),
                           ("methodology","METHODOLOGY"),("results","RESULTS")]:
            if sections.get(key):
                part = self.abstractive_summarize(sections[key], max_len=150, min_len=50)
                if part:
                    summary.append(f"{label}: {part}")
        return "\n\n".join(summary)

    def extract_key_insights(self, sections, text):
        insights = []
        source_text = (sections.get("conclusion") or sections.get("results") or text or "")
        for sent in re.split(r"(?<=[.!?])\s+", source_text):
            sent = self.clean_text(sent)
            if len(sent) > 60 and any(word in sent.lower() for word in 
                ["achieve","accuracy","improve","result","conclude","demonstrate","outperform"]):
                insights.append(sent)
                if len(insights) >= 10:
                    break
        return insights
