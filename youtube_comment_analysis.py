"""
YouTube Video Yorum Analizi - Stratejik Pazar Araştırması
Bu script bir YouTube videosunun yorumlarını analiz ederek pazar araştırması raporu oluşturur.
"""

import os
import re
import json
import time
from collections import Counter
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.util import bigrams
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# NLTK verilerini indir (gerekirse)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    try:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except:
        pass

# POS tagger için ek kontrol
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except:
        # Eğer indirilemezse alternatif yöntem deneyelim
        nltk.download('omw-1.4', quiet=True)
        try:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass

# Konfigürasyon
API_KEY = "AIzaSyC3X534_X2ad2ZKop6auc0rdjDEcrus1ME"
VIDEO_ID = "O0cs8aIXgkc"

# Stopwords
STOP_WORDS = set(stopwords.words('english'))

# Analiz kelime kümeleri
ECONOMIC_WORDS = ["buy", "price", "money", "worth", "expensive", "cheap", "economy", "pre-order"]
RISK_WORDS = ["fail", "broken", "ugly", "dangerous", "risk", "recall", "gap", "issue", "rust", "bad", "disaster"]
OPPORTUNITY_WORDS = ["future", "innovation", "tech", "growth", "best", "iconic", "change", "amazing", "love", "buy"]

# ============================================================================
# 1. VERİ TOPLAMA MODÜLÜ
# ============================================================================

def fetch_all_replies_for_thread(youtube, parent_id: str, max_results: int = None) -> List[Dict]:
    """
    Belirli bir thread'in TÜM yanıtlarını çeker (pagination ile).
    Basit ve direkt yaklaşım - karmaşık deduplication yok.
    
    Args:
        youtube: YouTube API service object
        parent_id: Ana yorumun ID'si (thread ID)
        max_results: Maksimum yanıt sayısı (None = sınırsız)
        
    Returns:
        Tüm yanıtların listesi (her yanıt dict formatında)
    """
    replies = []
    
    try:
        request = youtube.comments().list(
            part='snippet',
            parentId=parent_id,
            maxResults=100,  # API'nin izin verdiği maksimum
            textFormat='plainText'
        )
        
        while request:
            try:
                response = request.execute()
                
                # Yanıtları doğrudan ekle (deduplication yok)
                for item in response.get('items', []):
                    try:
                        reply_snippet = item['snippet']
                        replies.append({
                            'text': reply_snippet['textDisplay'],
                            'author': reply_snippet['authorDisplayName'],
                            'like_count': reply_snippet['likeCount'],
                            'published_at': reply_snippet['publishedAt'],
                            'is_reply': True
                        })
                        
                        # max_results kontrolü
                        if max_results is not None and len(replies) >= max_results:
                            break
                    except KeyError:
                        # Eksik verili yanıt, atla
                        continue
                
                # max_results kontrolü
                if max_results is not None and len(replies) >= max_results:
                    break
                
                # Sonraki sayfaya geç
                if 'nextPageToken' in response:
                    request = youtube.comments().list(
                        part='snippet',
                        parentId=parent_id,
                        maxResults=100,
                        textFormat='plainText',
                        pageToken=response['nextPageToken']
                    )
                else:
                    # Tüm yanıtlar çekildi
                    break
                    
            except HttpError as e:
                error_content = str(e)
                # Rate limiting veya quota hatası
                if '429' in error_content or 'quotaExceeded' in error_content:
                    break
                # Diğer hatalar için de durdur
                break
                    
            except Exception as e:
                # Diğer hatalar için durdur
                break
                
    except Exception as e:
        # Hata durumunda boş liste döndür (sessizce devam et)
        pass
    
    return replies


def fetch_youtube_comments(video_id: str, api_key: str, max_results: int = None) -> List[Dict]:
    """
    YouTube API kullanarak video yorumlarını çeker.
    KRİTİK: order='time' kullanarak tüm yorumları kronolojik sırada çeker.
    Basit ve direkt yaklaşım - karmaşık deduplication mantığı yok.
    
    Args:
        video_id: YouTube video ID
        api_key: YouTube Data API v3 key
        max_results: Maksimum çekilecek yorum sayısı (None = sınırsız, TÜM yorumlar)
        
    Returns:
        Yorum listesi (her yorum dict formatında)
    """
    comments = []
    top_level_count = 0  # Ana yorum sayacı
    reply_count = 0  # Yanıt sayacı
    last_progress_count = 0
    progress_interval = 500  # Her 500 yorumda bir progress göster
    
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # KRİTİK: order='time' kullan (relevance yerine)
        try:
            request = youtube.commentThreads().list(
                part='snippet',  # Sadece snippet - replies istemiyoruz
                videoId=video_id,
                maxResults=100,  # API'nin izin verdiği maksimum (100)
                textFormat='plainText',
                order='time'  # KRİTİK DEĞİŞİKLİK: relevance yerine time
            )
        except Exception as e:
            # order='time' bazen hata verebilir, relevance ile dene
            print(f"   ⚠️  order='time' hatası, 'relevance' deneniyor...")
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                textFormat='plainText',
                order='relevance'
            )
        
        page_count = 0
        retry_count = 0
        max_retries = 3
        
        print(f"🔄 Yorum çekme işlemi başlatıldı...")
        print(f"   📅 Sıralama: order='time' (kronolojik)")
        print(f"   🔍 Basit mod: Sadece snippet, karmaşık deduplication yok")
        if max_results is None:
            print(f"   ⚡ Sınırsız mod: TÜM yorumlar ve yanıtlar çekilecek")
        else:
            print(f"   📊 Maksimum {max_results} yorum çekilecek (ana yorumlar + yanıtlar dahil)")
        print()
        
        processed_threads = 0
        
        while request:
            try:
                # API isteği yap
                response = request.execute()
                page_items = response.get('items', [])
                
                # Her thread'i işle
                for item in page_items:
                    processed_threads += 1
                    try:
                        # 1. Ana yorumu ekle (doğrudan, karmaşık mantık yok)
                        top_level_comment = item['snippet']['topLevelComment']['snippet']
                        comments.append({
                            'text': top_level_comment['textDisplay'],
                            'author': top_level_comment['authorDisplayName'],
                            'like_count': top_level_comment['likeCount'],
                            'published_at': top_level_comment['publishedAt'],
                            'is_reply': False
                        })
                        top_level_count += 1
                        
                        # 2. Yanıtları çek (sadece totalReplyCount > 0 ise)
                        thread_id = item['snippet']['topLevelComment']['id']
                        total_reply_count = item['snippet'].get('totalReplyCount', 0)
                        
                        if total_reply_count > 0:
                            try:
                                # comments().list() ile tüm yanıtları çek
                                thread_replies = fetch_all_replies_for_thread(
                                    youtube, 
                                    thread_id, 
                                    max_results=None if max_results is None else (max_results - len(comments))
                                )
                                
                                # Yanıtları doğrudan ekle (deduplication yok)
                                for reply_dict in thread_replies:
                                    comments.append(reply_dict)
                                    reply_count += 1
                                    
                                    # max_results kontrolü
                                    if max_results is not None and len(comments) >= max_results:
                                        break
                                
                            except HttpError as e:
                                # API hatası - bu thread'i atla, devam et
                                error_str = str(e)
                                if '403' in error_str or '404' in error_str or '400' in error_str:
                                    # Sessizce atla
                                    pass
                            except Exception as e:
                                # Diğer hatalar - sessizce atla
                                pass
                        
                        # max_results kontrolü
                        if max_results is not None and len(comments) >= max_results:
                            break
                            
                    except KeyError as e:
                        # Bazı yorumlar eksik veri içerebilir, atla
                        continue
                
                # Progress göster
                if len(comments) - last_progress_count >= progress_interval:
                    print(f"   📥 Toplam: {len(comments):,} yorum ({top_level_count:,} ana + {reply_count:,} yanıt) | Thread: {processed_threads}")
                    last_progress_count = len(comments)
                
                # max_results kontrolü
                if max_results is not None and len(comments) >= max_results:
                    print(f"\n   ✓ Hedef sayıya ulaşıldı: {max_results} yorum")
                    break
                
                # Sonraki sayfaya geç
                if 'nextPageToken' in response:
                    try:
                        request = youtube.commentThreads().list(
                            part='snippet',  # Sadece snippet
                            videoId=video_id,
                            maxResults=100,
                            textFormat='plainText',
                            order='time',  # KRİTİK: time kullan
                            pageToken=response['nextPageToken']
                        )
                    except Exception:
                        # order='time' hatası, relevance ile dene
                        request = youtube.commentThreads().list(
                            part='snippet',
                            videoId=video_id,
                            maxResults=100,
                            textFormat='plainText',
                            order='relevance',
                            pageToken=response['nextPageToken']
                        )
                    page_count += 1
                    retry_count = 0
                    
                    # Rate limiting koruması
                    if page_count % 10 == 0:
                        time.sleep(0.1)
                else:
                    # Tüm yorumlar çekildi
                    print(f"\n   ✓ Tüm ana yorumlar çekildi (nextPageToken yok)")
                    break
                    
            except HttpError as e:
                error_content = str(e)
                
                # API kotası hatası
                if 'quotaExceeded' in error_content or '403' in error_content:
                    print(f"\n   ⚠️  API kotası doldu veya erişim hatası.")
                    print(f"   Mevcut {len(comments):,} yorumla devam ediliyor...")
                    break
                
                # Rate limiting hatası - retry dene
                elif '429' in error_content or 'rateLimitExceeded' in error_content:
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 saniye
                        print(f"\n   ⏳ Rate limit aşıldı. {wait_time} saniye bekleniyor... (Deneme {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                        continue  # Aynı request'i tekrar dene
                    else:
                        print(f"\n   ❌ Rate limit hatası devam ediyor. Mevcut {len(comments):,} yorumla devam ediliyor...")
                        break
                
                # Diğer HTTP hataları
                else:
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = 2 ** retry_count
                        print(f"\n   ⚠️  HTTP hatası: {error_content[:100]}")
                        print(f"   {wait_time} saniye beklenip tekrar denenecek... (Deneme {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"\n   ❌ HTTP hatası devam ediyor. Mevcut {len(comments):,} yorumla devam ediliyor...")
                        break
                        
            except Exception as e:
                # Network hatası, timeout vs.
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = 2 ** retry_count
                    print(f"\n   ⚠️  Bağlantı hatası: {str(e)[:100]}")
                    print(f"   {wait_time} saniye beklenip tekrar denenecek... (Deneme {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"\n   ❌ Bağlantı hatası devam ediyor. Mevcut {len(comments):,} yorumla devam ediliyor...")
                    break
                    
    except Exception as e:
        print(f"\n   ❌ API bağlantı hatası: {str(e)}")
        if len(comments) == 0:
            raise Exception("Yorum çekilemedi! Lütfen API key'i kontrol edin.")
        else:
            print(f"   Ancak {len(comments):,} yorum başarıyla çekilmiş durumda.")
    
    # Final progress mesajı
    if len(comments) > last_progress_count:
        print(f"\n   📥 Son durum: {len(comments):,} yorum ({top_level_count:,} ana + {reply_count:,} yanıt)")
    
    print(f"\n✅ Toplam {len(comments):,} yorum başarıyla çekildi ve analize hazır!")
    print(f"   📊 Detay: {top_level_count:,} ana yorum + {reply_count:,} yanıt")
    print(f"   🔄 İşlenen thread sayısı: {processed_threads}")
    print(f"   📄 İşlenen sayfa sayısı: {page_count}")
    
    return comments


# ============================================================================
# 2. VERİ TEMİZLEME MODÜLÜ
# ============================================================================

def clean_text(text: str) -> str:
    """
    Metni temizler: emoji, URL, kullanıcı etiketleri ve stopwords kaldırır.
    
    Args:
        text: Temizlenecek metin
        
    Returns:
        Temizlenmiş metin
    """
    if not isinstance(text, str):
        return ""
    
    # Emoji temizleme
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # URL temizleme
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Kullanıcı etiketleri temizleme (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Küçük harfe çevirme
    text = text.lower()
    
    # Özel karakterleri koruyarak temizleme (noktalama işaretleri analiz için gerekli olabilir)
    # Sadece fazladan boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords_from_text(text: str) -> str:
    """
    Metinden stopwords'leri kaldırır.
    
    Args:
        text: İşlenecek metin
        
    Returns:
        Stopwords'lerden temizlenmiş metin
    """
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in STOP_WORDS and token.isalpha()]
    return ' '.join(filtered_tokens)


# ============================================================================
# 3. SENTIMENT ANALİZİ MODÜLÜ
# ============================================================================

def classify_sentiment(text: str) -> str:
    """
    TextBlob kullanarak sentiment sınıflandırması yapar.
    
    Args:
        text: Analiz edilecek metin
        
    Returns:
        'Positive', 'Negative' veya 'Neutral'
    """
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    except:
        return 'Neutral'


def analyze_sentiment(comments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm yorumlara sentiment analizi uygular.
    
    Args:
        comments_df: Yorumlar dataframe'i
        
    Returns:
        Sentiment kolonu eklenmiş dataframe
    """
    print("📊 Sentiment analizi yapılıyor...")
    comments_df['sentiment'] = comments_df['cleaned_text'].apply(classify_sentiment)
    return comments_df


def economic_confidence_analysis(comments_df: pd.DataFrame) -> Dict:
    """
    Ekonomik kelimeler içeren yorumlarda sentiment analizi yapar.
    
    Args:
        comments_df: Yorumlar dataframe'i
        
    Returns:
        Analiz sonuçları dictionary
    """
    print("💰 Ekonomik güven analizi yapılıyor...")
    
    # Ekonomik kelimeler içeren yorumları filtrele
    economic_pattern = '|'.join(ECONOMIC_WORDS)
    economic_comments = comments_df[comments_df['cleaned_text'].str.contains(economic_pattern, case=False, na=False)]
    
    if len(economic_comments) == 0:
        print("⚠️  Ekonomik kelime içeren yorum bulunamadı!")
        return {
            'total': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'confidence_score': 0.0,
            'distribution': {}
        }
    
    # Sentiment dağılımı
    sentiment_dist = economic_comments['sentiment'].value_counts().to_dict()
    
    positive_count = sentiment_dist.get('Positive', 0)
    negative_count = sentiment_dist.get('Negative', 0)
    neutral_count = sentiment_dist.get('Neutral', 0)
    total_count = len(economic_comments)
    
    # Güven skoru: (Positive - Negative) / Total
    confidence_score = (positive_count - negative_count) / total_count if total_count > 0 else 0.0
    
    print(f"   Ekonomik yorum sayısı: {total_count}")
    print(f"   Positive: {positive_count} ({positive_count/total_count*100:.1f}%)")
    print(f"   Negative: {negative_count} ({negative_count/total_count*100:.1f}%)")
    print(f"   Güven Skoru: {confidence_score:.3f}")
    
    return {
        'total': total_count,
        'positive': positive_count,
        'negative': negative_count,
        'neutral': neutral_count,
        'confidence_score': confidence_score,
        'distribution': {
            'Positive': positive_count / total_count * 100,
            'Negative': negative_count / total_count * 100,
            'Neutral': neutral_count / total_count * 100
        }
    }


# ============================================================================
# 4. RİSK VS. FIRSAT ANALİZİ MODÜLÜ
# ============================================================================

def risk_opportunity_analysis(comments_df: pd.DataFrame) -> Dict:
    """
    Risk ve fırsat kelimelerinin co-occurrence analizini yapar.
    
    Args:
        comments_df: Yorumlar dataframe'i
        
    Returns:
        Analiz sonuçları dictionary
    """
    print("🎯 Risk vs. Fırsat analizi yapılıyor...")
    
    risk_frequencies = Counter()
    opportunity_frequencies = Counter()
    
    # Marka kelimeleri
    brand_words = ['tesla', 'cybertruck', 'elon', 'musk']
    
    for idx, row in comments_df.iterrows():
        text = row['cleaned_text']
        text_lower = text.lower()
        
        # Risk kelimelerini say
        for word in RISK_WORDS:
            if word in text_lower:
                # Marka kelimeleriyle birlikte geçiyor mu?
                has_brand = any(brand in text_lower for brand in brand_words)
                risk_frequencies[word] += 1
                if has_brand:
                    risk_frequencies[f"{word}_with_brand"] += 1
        
        # Fırsat kelimelerini say
        for word in OPPORTUNITY_WORDS:
            if word in text_lower:
                has_brand = any(brand in text_lower for brand in brand_words)
                opportunity_frequencies[word] += 1
                if has_brand:
                    opportunity_frequencies[f"{word}_with_brand"] += 1
    
    total_risk = sum([v for k, v in risk_frequencies.items() if not k.endswith('_with_brand')])
    total_opportunity = sum([v for k, v in opportunity_frequencies.items() if not k.endswith('_with_brand')])
    
    risk_with_brand = sum([v for k, v in risk_frequencies.items() if k.endswith('_with_brand')])
    opportunity_with_brand = sum([v for k, v in opportunity_frequencies.items() if k.endswith('_with_brand')])
    
    print(f"   Risk kelime toplamı: {total_risk}")
    print(f"   Fırsat kelime toplamı: {total_opportunity}")
    print(f"   Risk+Marka: {risk_with_brand}, Fırsat+Marka: {opportunity_with_brand}")
    
    return {
        'risk_frequencies': dict(risk_frequencies),
        'opportunity_frequencies': dict(opportunity_frequencies),
        'total_risk': total_risk,
        'total_opportunity': total_opportunity,
        'risk_with_brand': risk_with_brand,
        'opportunity_with_brand': opportunity_with_brand,
        'risk_words': RISK_WORDS,
        'opportunity_words': OPPORTUNITY_WORDS
    }


# ============================================================================
# 5. KORKU MADENCİLİĞİ (FEAR MINING) MODÜLÜ
# ============================================================================

def fear_mining_analysis(comments_df: pd.DataFrame) -> List[Tuple[str, int]]:
    """
    Negatif yorumlardan bigram çıkararak ana korkuları tespit eder.
    
    Args:
        comments_df: Yorumlar dataframe'i
        
    Returns:
        En sık geçen bigram listesi (kelime çifti, frekans)
    """
    print("😨 Korku madenciliği (Fear Mining) analizi yapılıyor...")
    
    # Negatif yorumları filtrele
    negative_comments = comments_df[comments_df['sentiment'] == 'Negative']
    
    if len(negative_comments) == 0:
        print("⚠️  Negatif yorum bulunamadı!")
        return []
    
    print(f"   Negatif yorum sayısı: {len(negative_comments)}")
    
    # Tüm negatif yorumları birleştir
    all_text = ' '.join(negative_comments['cleaned_text'].tolist())
    
    # Tokenize et
    tokens = word_tokenize(all_text)
    
    # POS tagging yap (hata yönetimi ile)
    try:
        pos_tags = pos_tag(tokens)
    except LookupError as e:
        print(f"   ⚠️  POS tagger hatası: {str(e)}")
        print("   NLTK tagger verileri indiriliyor...")
        try:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            pos_tags = pos_tag(tokens)
        except:
            try:
                nltk.download('averaged_perceptron_tagger', quiet=True)
                pos_tags = pos_tag(tokens)
            except Exception as e2:
                print(f"   ❌ POS tagging yapılamadı: {str(e2)}")
                print("   Alternatif yöntem kullanılıyor: basit bigram analizi...")
                # Alternatif: Basit bigram analizi (POS tagging olmadan)
                bigram_list = []
                for i in range(len(tokens) - 1):
                    word1 = tokens[i].lower()
                    word2 = tokens[i + 1].lower()
                    if word1.isalpha() and word2.isalpha():
                        if word1 not in STOP_WORDS and word2 not in STOP_WORDS:
                            bigram = f"{word1} {word2}"
                            bigram_list.append(bigram)
                bigram_counter = Counter(bigram_list)
                top_bigrams = bigram_counter.most_common(20)
                print(f"   Tespit edilen unique bigram sayısı: {len(bigram_counter)}")
                for bigram, count in top_bigrams[:10]:
                    print(f"      '{bigram}': {count} kez")
                return top_bigrams
    
    # Noun ve Adjective'leri filtrele
    noun_adj_pairs = []
    for i in range(len(pos_tags) - 1):
        word1, tag1 = pos_tags[i]
        word2, tag2 = pos_tags[i + 1]
        
        # Sadece alfanumerik kelimeler
        if not (word1.isalpha() and word2.isalpha()):
            continue
        
        # Stopwords değilse
        if word1.lower() in STOP_WORDS or word2.lower() in STOP_WORDS:
            continue
        
        # Noun-Noun, Adjective-Noun, Noun-Adjective kombinasyonları
        if ((tag1.startswith('NN') and tag2.startswith('NN')) or
            (tag1.startswith('JJ') and tag2.startswith('NN')) or
            (tag1.startswith('NN') and tag2.startswith('JJ'))):
            bigram = f"{word1.lower()} {word2.lower()}"
            noun_adj_pairs.append(bigram)
    
    # Bigram frekanslarını say
    bigram_counter = Counter(noun_adj_pairs)
    
    # En sık geçen 20 bigram'ı al
    top_bigrams = bigram_counter.most_common(20)
    
    print(f"   Tespit edilen unique bigram sayısı: {len(bigram_counter)}")
    print(f"   En sık geçen bigram'lar:")
    for bigram, count in top_bigrams[:10]:
        print(f"      '{bigram}': {count} kez")
    
    return top_bigrams


# ============================================================================
# 6. GÖRSELLEŞTİRME MODÜLÜ
# ============================================================================

def create_visualizations(comments_df: pd.DataFrame, 
                         economic_results: Dict,
                         risk_opp_results: Dict,
                         fear_bigrams: List[Tuple[str, int]]) -> None:
    """
    Tüm görselleştirmeleri oluşturur ve kaydeder.
    
    Args:
        comments_df: Yorumlar dataframe'i
        economic_results: Ekonomik güven analizi sonuçları
        risk_opp_results: Risk-fırsat analizi sonuçları
        fear_bigrams: Korku bigram'ları
    """
    print("📈 Görselleştirmeler oluşturuluyor...")
    
    # Stil ayarları
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Genel Duygu Dağılımı (Pasta Grafiği)
    sentiment_counts = comments_df['sentiment'].value_counts()
    
    # Renk haritası - Sabit renkler
    color_map = {
        'Positive': '#2ecc71',  # Yeşil
        'Negative': '#e74c3c',  # Kırmızı
        'Neutral': '#95a5a6'    # Gri
    }
    
    # Mevcut sentimentlere göre renk listesi oluştur
    # value_counts() sırasına göre renkleri eşleştir
    colors = [color_map.get(label, '#95a5a6') for label in sentiment_counts.index]
    
    plt.figure(figsize=(10, 8))
    plt.pie(sentiment_counts.values, 
            labels=sentiment_counts.index, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title('Genel Duygu Dağılımı\n(Tüm Yorumlar)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ sentiment_distribution.png kaydedildi (Renkler düzeltildi)")
    
    # 2. Risk Kelime Bulutu
    if risk_opp_results['total_risk'] > 0:
        risk_text = ' '.join([word for word in RISK_WORDS for _ in range(risk_opp_results['risk_frequencies'].get(word, 0))])
        if risk_text.strip():
            plt.figure(figsize=(16, 10))  # Daha büyük boyut
            wordcloud = WordCloud(width=1600, height=1000, 
                                background_color='white',
                                colormap='Reds',
                                max_words=200,  # Daha fazla kelime
                                relative_scaling=0.5, # Kelime sıklığına göre ölçekleme
                                prefer_horizontal=0.7).generate(risk_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Risk Kelime Bulutu', fontsize=20, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('risk_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✅ risk_wordcloud.png kaydedildi (Daha detaylı)")
    
    # 3. Fırsat Kelime Bulutu
    if risk_opp_results['total_opportunity'] > 0:
        opp_text = ' '.join([word for word in OPPORTUNITY_WORDS for _ in range(risk_opp_results['opportunity_frequencies'].get(word, 0))])
        if opp_text.strip():
            plt.figure(figsize=(16, 10))  # Daha büyük boyut
            wordcloud = WordCloud(width=1600, height=1000, 
                                background_color='white',
                                colormap='Greens',
                                max_words=200,  # Daha fazla kelime
                                relative_scaling=0.5,
                                prefer_horizontal=0.7).generate(opp_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Fırsat Kelime Bulutu', fontsize=20, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('opportunity_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✅ opportunity_wordcloud.png kaydedildi (Daha detaylı)")
    
    # 4. Ekonomik Güven Skoru (Bar Grafiği)
    if economic_results['total'] > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Positive', 'Negative', 'Neutral']
        percentages = [
            economic_results['distribution'].get('Positive', 0),
            economic_results['distribution'].get('Negative', 0),
            economic_results['distribution'].get('Neutral', 0)
        ]
        colors_bar = ['#2ecc71', '#e74c3c', '#95a5a6']
        bars = ax.bar(categories, percentages, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Yüzdeleri çubukların üzerine yaz
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{pct:.1f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Yüzde (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sentiment Kategorisi', fontsize=12, fontweight='bold')
        ax.set_title(f'Ekonomik Güven Skoru\n(Ekonomik Kelimeler İçeren Yorumlar)\nGüven Skoru: {economic_results["confidence_score"]:.3f}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim([0, max(percentages) * 1.2 + 10])
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('economic_confidence.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ economic_confidence.png kaydedildi")


# ============================================================================
# 7. RAPOR OLUŞTURMA MODÜLÜ
# ============================================================================

def generate_report(comments_df: pd.DataFrame,
                   economic_results: Dict,
                   risk_opp_results: Dict,
                   fear_bigrams: List[Tuple[str, int]]) -> None:
    """
    Markdown formatında stratejik rapor oluşturur.
    
    Args:
        comments_df: Yorumlar dataframe'i
        economic_results: Ekonomik güven analizi sonuçları
        risk_opp_results: Risk-fırsat analizi sonuçları
        fear_bigrams: Korku bigram'ları
    """
    print("📝 Rapor oluşturuluyor...")
    
    total_comments = len(comments_df)
    sentiment_dist = comments_df['sentiment'].value_counts().to_dict()
    
    report = f"""# YouTube Video Yorum Analizi - Stratejik Pazar Araştırması Raporu

## 📊 Genel Bilgiler

- **Video ID**: {VIDEO_ID}
- **Video URL**: https://www.youtube.com/watch?v={VIDEO_ID}
- **Analiz Edilen Toplam Yorum Sayısı**: {total_comments}
- **Genel Sentiment Dağılımı**:
  - Positive: {sentiment_dist.get('Positive', 0)} ({sentiment_dist.get('Positive', 0)/total_comments*100:.1f}%)
  - Negative: {sentiment_dist.get('Negative', 0)} ({sentiment_dist.get('Negative', 0)/total_comments*100:.1f}%)
  - Neutral: {sentiment_dist.get('Neutral', 0)} ({sentiment_dist.get('Neutral', 0)/total_comments*100:.1f}%)

---

## ❓ Soru 1: Psikolojik Atmosfer (Tüketici Güveni) Analizi

### Araştırma Sorusu
Ekonomik terimler içeren yorumlarda tüketici güveni pozitif mi, negatif mi?

### Metodoloji
- Yorumlar TextBlob kullanılarak Positive, Negative ve Neutral olarak sınıflandırıldı.
- Ekonomik kelimeler içeren yorumlar filtrelandı: `{', '.join(ECONOMIC_WORDS)}`
- Ekonomik Güven Skoru hesaplandı: (Positive Count - Negative Count) / Total Count

### Veriye Dayalı Bulgular

**Ekonomik Kelime İçeren Yorum Sayısı**: {economic_results['total']}

**Sentiment Dağılımı**:
- **Positive**: {economic_results['positive']} yorum ({economic_results['distribution'].get('Positive', 0):.1f}%)
- **Negative**: {economic_results['negative']} yorum ({economic_results['distribution'].get('Negative', 0):.1f}%)
- **Neutral**: {economic_results['neutral']} yorum ({economic_results['distribution'].get('Neutral', 0):.1f}%)

**Ekonomik Güven Skoru**: `{economic_results['confidence_score']:.3f}`

### Yorum ve Sonuç

"""

    # Güven skoru yorumu
    if economic_results['confidence_score'] > 0.2:
        report += "✅ **Tüketici güveni YÜKSEK seviyede**. Ekonomik terimler içeren yorumlarda pozitif sentiment baskın durumda. "
        report += f"Yorumların %{economic_results['distribution'].get('Positive', 0):.1f}'i satın alma, fiyat ve değer konusunda olumlu görüşler içeriyor. "
        report += "Bu, tüketicilerin ürünün değerine olan inancını ve satın alma niyetini gösteriyor.\n\n"
    elif economic_results['confidence_score'] < -0.2:
        report += "❌ **Tüketici güveni DÜŞÜK seviyede**. Ekonomik terimler içeren yorumlarda negatif sentiment baskın durumda. "
        report += f"Yorumların %{economic_results['distribution'].get('Negative', 0):.1f}'i fiyat, maliyet ve değer konusunda endişeler içeriyor. "
        report += "Bu, tüketicilerin ürünün fiyat/değer oranından memnun olmadığını ve satın alma niyetlerinin düşük olduğunu gösteriyor.\n\n"
    else:
        report += "⚖️ **Tüketici güveni ORTA seviyede**. Ekonomik terimler içeren yorumlarda pozitif ve negatif görüşler dengeli. "
        report += "Tüketicilerin bir kısmı ürünün değerli olduğunu düşünürken, diğer kısmı fiyat/değer konusunda tereddütlü. "
        report += "Bu, pazar segmentasyonu ve hedefli pazarlama stratejileri geliştirilmesi gerektiğini gösteriyor.\n\n"

    report += """---

## ❓ Soru 2: Kriz/Risk vs. Fırsat/Büyüme Eşdizimliliği (Co-occurrence)

### Araştırma Sorusu
Marka (Tesla/Cybertruck) en çok hangi küme ile (Risk mi, Fırsat mı) yan yana geliyor?

### Metodoloji
- **Risk Kelime Kümesi**: `""" + ', '.join(RISK_WORDS) + """`
- **Fırsat Kelime Kümesi**: `""" + ', '.join(OPPORTUNITY_WORDS) + """`
- Yorumlarda bu kelimelerin frekansları sayıldı.
- Marka kelimeleriyle (Tesla, Cybertruck, Elon, Musk) birlikte geçen kelimeler tespit edildi.

### Veriye Dayalı Bulgular

**Toplam Risk Kelime Kullanımı**: {risk_opp_results['total_risk']} kez
**Toplam Fırsat Kelime Kullanımı**: {risk_opp_results['total_opportunity']} kez

**Marka ile Birlikte Geçen Kelimeler**:
- **Risk Kelimeleri + Marka**: {risk_opp_results['risk_with_brand']} kez
- **Fırsat Kelimeleri + Marka**: {risk_opp_results['opportunity_with_brand']} kez

**Risk/Fırsat Oranı**: {risk_opp_results['total_risk']/risk_opp_results['total_opportunity']:.2f} (Risk kelimeleri, fırsat kelimelerinin {risk_opp_results['total_risk']/risk_opp_results['total_opportunity']:.2f} katı)

"""

    # En sık kullanılan risk kelimeleri
    if risk_opp_results['risk_frequencies']:
        top_risk = sorted([(k, v) for k, v in risk_opp_results['risk_frequencies'].items() 
                          if not k.endswith('_with_brand')], 
                         key=lambda x: x[1], reverse=True)[:5]
        report += "**En Sık Kullanılan Risk Kelimeleri**:\n"
        for word, count in top_risk:
            report += f"- `{word}`: {count} kez\n"
        report += "\n"

    # En sık kullanılan fırsat kelimeleri
    if risk_opp_results['opportunity_frequencies']:
        top_opp = sorted([(k, v) for k, v in risk_opp_results['opportunity_frequencies'].items() 
                         if not k.endswith('_with_brand')], 
                        key=lambda x: x[1], reverse=True)[:5]
        report += "**En Sık Kullanılan Fırsat Kelimeleri**:\n"
        for word, count in top_opp:
            report += f"- `{word}`: {count} kez\n"
        report += "\n"

    # Sonuç yorumu
    if risk_opp_results['risk_with_brand'] > risk_opp_results['opportunity_with_brand']:
        report += "### Yorum ve Sonuç\n\n"
        report += "⚠️ **Marka, Risk kelimeleriyle daha sık yan yana geliyor**. "
        report += f"Marka adı geçen yorumlarda risk kelimeleri {risk_opp_results['risk_with_brand']} kez, "
        report += f"fırsat kelimeleri ise {risk_opp_results['opportunity_with_brand']} kez kullanılmış. "
        report += "Bu, markanın tüketicilerin gözünde risk algısıyla daha çok ilişkilendirildiğini gösteriyor. "
        report += "Kalite sorunları, güvenlik endişeleri veya fiyat konusundaki şikayetler markayla özdeşleşmiş durumda.\n\n"
    elif risk_opp_results['opportunity_with_brand'] > risk_opp_results['risk_with_brand']:
        report += "### Yorum ve Sonuç\n\n"
        report += "✅ **Marka, Fırsat kelimeleriyle daha sık yan yana geliyor**. "
        report += f"Marka adı geçen yorumlarda fırsat kelimeleri {risk_opp_results['opportunity_with_brand']} kez, "
        report += f"risk kelimeleri ise {risk_opp_results['risk_with_brand']} kez kullanılmış. "
        report += "Bu, markanın tüketicilerin gözünde yenilik, büyüme ve olumlu değişimle ilişkilendirildiğini gösteriyor. "
        report += "Marka imajı pozitif ve gelecek odaklı bir algıya sahip.\n\n"
    else:
        report += "### Yorum ve Sonuç\n\n"
        report += "⚖️ **Marka, Risk ve Fırsat kelimeleriyle dengeli bir şekilde yan yana geliyor**. "
        report += "Bu durum, markanın hem olumlu hem de olumsuz yönleriyle tartışıldığını gösteriyor. "
        report += "Pazarın markayı dengeli bir şekilde değerlendirdiği söylenebilir.\n\n"

    report += """---

## ❓ Soru 3: Ana Korku Tespiti (Fear Mining)

### Araştırma Sorusu
Negatif yorumlarda en sık geçen isim-sıfat (Noun-Adjective) bigram'ları nelerdir? Tüketicilerin ana korkuları nedir?

### Metodoloji
- Sadece `Negative` sentiment'li yorumlar analiz edildi.
- NLTK POS Tagger kullanılarak isim (Noun) ve sıfat (Adjective) kelimeler tespit edildi.
- Noun-Noun, Adjective-Noun ve Noun-Adjective bigram'ları çıkarıldı.
- En sık geçen bigram'lar frekansına göre sıralandı.

### Veriye Dayalı Bulgular

**Analiz Edilen Negatif Yorum Sayısı**: """ + str(len(comments_df[comments_df['sentiment'] == 'Negative'])) + """

**En Sık Geçen Bigram'lar (Ana Korkular)**:\n\n"""

    if fear_bigrams:
        report += "| Sıra | Bigram | Frekans |\n"
        report += "|------|--------|---------|\n"
        for i, (bigram, count) in enumerate(fear_bigrams[:15], 1):
            report += f"| {i} | `{bigram}` | {count} |\n"
    else:
        report += "*Yeterli negatif yorum bulunamadı veya bigram çıkarılamadı.*\n"

    report += "\n### Yorum ve Sonuç\n\n"

    if fear_bigrams:
        top_fears = [bigram for bigram, _ in fear_bigrams[:5]]
        report += f"🔍 **Tüketicilerin Ana Korkuları**: Analiz sonucunda, negatif yorumlarda en sık geçen korku temaları şunlardır:\n\n"
        for i, fear in enumerate(top_fears, 1):
            report += f"{i}. **`{fear}`**: Bu bigram, tüketicilerin ürün/hizmet hakkındaki temel endişelerinden birini yansıtıyor.\n"
        
        report += "\nBu korkular, ürün geliştirme, kalite kontrolü ve pazarlama stratejilerinde dikkate alınması gereken kritik noktaları işaret ediyor. "
        report += "Özellikle bu temalar etrafında oluşturulacak içerikler ve iletişim stratejileri, tüketici endişelerini gidermeye yönelik olmalıdır.\n\n"
    else:
        report += "⚠️ Yeterli negatif yorum bulunamadığı için fear mining analizi yapılamadı. "
        report += "Bu durum, genel olarak yorumların pozitif veya nötr olduğunu gösterebilir.\n\n"

    report += """---

## 📈 Görselleştirmeler

Analiz sonuçları aşağıdaki görselleştirmelerde özetlenmiştir:

1. **sentiment_distribution.png**: Genel duygu dağılımı (Pasta Grafiği)
2. **risk_wordcloud.png**: Risk kelimeleri word cloud
3. **opportunity_wordcloud.png**: Fırsat kelimeleri word cloud
4. **economic_confidence.png**: Ekonomik güven skoru (Bar Grafiği)

---

## 🔍 Metodolojik Notlar

- **Sentiment Analizi**: TextBlob kütüphanesi kullanılarak polarity score hesaplanmıştır.
- **Veri Temizleme**: Emoji, URL, kullanıcı etiketleri ve stopwords temizlenmiştir.
- **Co-occurrence**: Kelime frekansları ve marka ile ilişkileri analiz edilmiştir.
- **Fear Mining**: NLTK POS tagger kullanılarak bigram çıkarımı yapılmıştır.

---

*Rapor, YouTube Data API v3 kullanılarak çekilen yorumlar üzerinde gerçekleştirilmiştir.*
*Tarih: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """*

"""

    # Raporu kaydet
    with open('analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("   ✅ analysis_report.md kaydedildi")


# ============================================================================
# 8. ANA FONKSİYON
# ============================================================================

def main():
    """
    Ana çalıştırma fonksiyonu.
    """
    print("=" * 60)
    print("YouTube Video Yorum Analizi - Pazar Araştırması")
    print("=" * 60)
    print()
    
    try:
        # 1. Yorumları çek (max_results=None ile tüm yorumları çek)
        print("📥 YouTube yorumları çekiliyor...")
        print("   (Tüm yorumlar çekilecek, bu işlem biraz zaman alabilir...)")
        comments = fetch_youtube_comments(VIDEO_ID, API_KEY, max_results=None)
        
        if len(comments) == 0:
            print("❌ Yorum çekilemedi! Program sonlandırılıyor.")
            return
        
        # 2. DataFrame oluştur
        comments_df = pd.DataFrame(comments)
        
        # 3. Metin temizleme
        print("🧹 Metinler temizleniyor...")
        comments_df['cleaned_text'] = comments_df['text'].apply(clean_text)
        comments_df['cleaned_text_no_stopwords'] = comments_df['cleaned_text'].apply(remove_stopwords_from_text)
        
        # Boş yorumları kaldır
        comments_df = comments_df[comments_df['cleaned_text'].str.len() > 0]
        print(f"   {len(comments_df)} yorum analiz için hazır")
        
        # 4. Sentiment analizi
        comments_df = analyze_sentiment(comments_df)
        
        # 5. Analizleri çalıştır
        economic_results = economic_confidence_analysis(comments_df)
        risk_opp_results = risk_opportunity_analysis(comments_df)
        fear_bigrams = fear_mining_analysis(comments_df)
        
        # 6. Görselleştirmeleri oluştur
        create_visualizations(comments_df, economic_results, risk_opp_results, fear_bigrams)
        
        # 7. Rapor oluştur
        generate_report(comments_df, economic_results, risk_opp_results, fear_bigrams)
        
        print()
        print("=" * 60)
        print("✅ Analiz tamamlandı!")
        print("=" * 60)
        print("\nOluşturulan dosyalar:")
        print("  📊 sentiment_distribution.png")
        print("  📊 risk_wordcloud.png")
        print("  📊 opportunity_wordcloud.png")
        print("  📊 economic_confidence.png")
        print("  📝 analysis_report.md")
        print()
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
