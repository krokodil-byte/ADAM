#!/usr/bin/env python3
"""
Wikipedia Training Module
Training su dump Wikipedia (XML o estratti) con fallback API
"""

import re
import json
import sys
import gc
from pathlib import Path
from typing import Optional, Iterator, List, Tuple
import time
import urllib.request
import urllib.parse

# Handle imports for both installed package and direct execution
try:
    from core.brain_wrapper import VectLLMBrain
    from core.stats import StatsCollector
    from Utils.checkpoint import CheckpointManager
except ImportError:
    from ..core.brain_wrapper import VectLLMBrain
    from ..core.stats import StatsCollector
    from ..utils.checkpoint import CheckpointManager


def get_memory_usage_percent() -> float:
    """Get current memory usage percentage without psutil"""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()

        meminfo = {}
        for line in lines:
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().split()[0]
                meminfo[key] = int(value)

        total = meminfo.get('MemTotal', 1)
        available = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
        used = total - available
        return (used / total) * 100
    except:
        return 0.0


class WikipediaAPIFetcher:
    """Fetch articles from Wikipedia API with RAM management"""

    def __init__(self,
                 language: str = 'en',
                 ram_percentage: float = 30.0,
                 min_article_length: int = 500,
                 max_article_length: int = 50000):
        """
        Args:
            language: Wikipedia language code (en, it, de, etc.)
            ram_percentage: Target RAM usage percentage for article buffer
            min_article_length: Minimum article length in chars
            max_article_length: Maximum article length in chars
        """
        self.language = language
        self.ram_percentage = ram_percentage
        self.min_length = min_article_length
        self.max_length = max_article_length
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"

        # Article buffer
        self.articles: List[Tuple[str, str]] = []
        self.total_fetched = 0

        print(f"📡 Wikipedia API Fetcher initialized")
        print(f"   Language: {language}")
        print(f"   RAM target: {ram_percentage}%")

    def _api_request(self, params: dict) -> dict:
        """Make API request to Wikipedia"""
        params['format'] = 'json'
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"

        try:
            # Wikipedia API requires a proper User-Agent
            request = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'A.D.A.M/1.0 (Adaptive Dynamic Agent Module; https://github.com/krokodil-byte/A.D.A.M) Python/urllib'
                }
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"   ⚠️  API error: {e}")
            return {}

    def _get_random_titles(self, count: int = 10) -> List[str]:
        """Get random article titles"""
        params = {
            'action': 'query',
            'list': 'random',
            'rnnamespace': 0,  # Main namespace only
            'rnlimit': count
        }

        result = self._api_request(params)
        if 'query' in result and 'random' in result['query']:
            return [item['title'] for item in result['query']['random']]
        return []

    def _get_article_content(self, title: str) -> Optional[str]:
        """Get article content by title"""
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain'
        }

        result = self._api_request(params)
        if 'query' in result and 'pages' in result['query']:
            pages = result['query']['pages']
            for page_id, page in pages.items():
                if page_id != '-1' and 'extract' in page:
                    return page['extract']
        return None

    def clean_text(self, text: str) -> str:
        """Clean Wikipedia text"""
        # Remove section markers
        text = re.sub(r'==+\s*[^=]+\s*==+', '\n', text)

        # Clean whitespace
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()

        return text

    def fetch_batch(self, batch_size: int = 50) -> int:
        """
        Fetch a batch of articles until RAM target is reached.

        Args:
            batch_size: Number of titles to fetch per API call

        Returns:
            Number of articles fetched in this batch
        """
        fetched = 0
        initial_memory = get_memory_usage_percent()
        target_memory = initial_memory + self.ram_percentage

        print(f"\n📥 Fetching articles (RAM: {initial_memory:.1f}% → {target_memory:.1f}%)")

        while get_memory_usage_percent() < target_memory:
            titles = self._get_random_titles(batch_size)

            if not titles:
                print("   ⚠️  No titles received, retrying...")
                time.sleep(1)
                continue

            for title in titles:
                content = self._get_article_content(title)

                if content:
                    content = self.clean_text(content)

                    if self.min_length <= len(content) <= self.max_length:
                        self.articles.append((title, content))
                        fetched += 1
                        self.total_fetched += 1

                        if fetched % 10 == 0:
                            mem = get_memory_usage_percent()
                            print(f"   Fetched {fetched} articles (RAM: {mem:.1f}%)")

                # Check RAM after each article
                if get_memory_usage_percent() >= target_memory:
                    break

            # Small delay to be nice to Wikipedia API
            time.sleep(0.5)

        print(f"   ✓ Batch complete: {fetched} articles")
        return fetched

    def get_articles(self) -> List[Tuple[str, str]]:
        """Get current article buffer"""
        return self.articles

    def clear_buffer(self):
        """Clear article buffer and free memory"""
        self.articles = []
        gc.collect()
        print(f"   🧹 Buffer cleared (RAM: {get_memory_usage_percent():.1f}%)")

    def iter_articles(self) -> Iterator[Tuple[str, str]]:
        """Iterate over buffered articles"""
        for title, text in self.articles:
            yield title, text


class WikipediaExtractor:
    """Estrae testo pulito da Wikipedia dump"""
    
    def __init__(self, 
                 dump_path: str,
                 min_article_length: int = 100,
                 max_article_length: int = 100000):
        """
        Args:
            dump_path: Path al dump Wikipedia (XML o JSONL)
            min_article_length: Lunghezza minima articolo (chars)
            max_article_length: Lunghezza massima articolo (chars)
        """
        self.dump_path = Path(dump_path)
        self.min_length = min_article_length
        self.max_length = max_article_length
        
        if not self.dump_path.exists():
            raise FileNotFoundError(f"Dump not found: {dump_path}")
        
        # Detect format
        self.format = self._detect_format()
        print(f"📚 Wikipedia dump format: {self.format}")
    
    def _detect_format(self) -> str:
        """Rileva formato dump"""
        ext = self.dump_path.suffix.lower()
        
        if ext in ['.xml', '.bz2']:
            return 'xml'
        elif ext in ['.json', '.jsonl']:
            return 'jsonl'
        elif ext in ['.txt']:
            return 'text'
        else:
            return 'unknown'
    
    def clean_text(self, text: str) -> str:
        """
        Pulisce testo Wikipedia da markup.
        
        Args:
            text: Testo raw
            
        Returns:
            Testo pulito
        """
        # Remove wiki links [[link|text]] -> text
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
        
        # Remove external links [url text] -> text
        text = re.sub(r'\[https?://[^\s\]]+ ([^\]]+)\]', r'\1', text)
        
        # Remove templates {{template}}
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        
        # Remove html tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove wiki formatting
        text = re.sub(r"'''([^']+)'''", r'\1', text)  # Bold
        text = re.sub(r"''([^']+)''", r'\1', text)     # Italic
        
        # Remove references
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*/>', '', text)
        
        # Clean whitespace
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text
    
    def iter_articles_jsonl(self) -> Iterator[tuple]:
        """
        Itera articoli da formato JSONL.
        
        Yields:
            (title, text)
        """
        with open(self.dump_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    article = json.loads(line)
                    title = article.get('title', f'Article {line_num}')
                    text = article.get('text', '')
                    
                    # Clean and filter
                    text = self.clean_text(text)
                    
                    if self.min_length <= len(text) <= self.max_length:
                        yield title, text
                        
                except json.JSONDecodeError:
                    continue
    
    def iter_articles_text(self) -> Iterator[tuple]:
        """
        Itera articoli da file di testo semplice.
        Assume formato: === Title ===\\nContent\\n\\n
        
        Yields:
            (title, text)
        """
        with open(self.dump_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by section markers
        articles = re.split(r'\n===\s*([^=]+)\s*===\n', content)
        
        for i in range(1, len(articles), 2):
            if i + 1 < len(articles):
                title = articles[i].strip()
                text = articles[i + 1].strip()
                
                if self.min_length <= len(text) <= self.max_length:
                    yield title, text
    
    def iter_articles(self) -> Iterator[tuple]:
        """
        Itera articoli (auto-detect format).
        
        Yields:
            (title, text)
        """
        if self.format == 'jsonl':
            yield from self.iter_articles_jsonl()
        elif self.format == 'text':
            yield from self.iter_articles_text()
        else:
            raise NotImplementedError(f"Format {self.format} not yet supported")


class WikipediaTrainer:
    """Training su Wikipedia con monitoring avanzato"""
    
    def __init__(self,
                 brain: VectLLMBrain,
                 extractor: WikipediaExtractor,
                 stats_collector: Optional[StatsCollector] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None):
        """
        Args:
            brain: Istanza VectLLMBrain
            extractor: Extractor Wikipedia
            stats_collector: Collector statistiche
            checkpoint_manager: Manager checkpoint
        """
        self.brain = brain
        self.extractor = extractor
        self.stats = stats_collector or StatsCollector()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
    
    def train(self,
              max_articles: Optional[int] = None,
              auto_save_every: int = 100,
              verbose: bool = True) -> dict:
        """
        Train su Wikipedia.
        
        Args:
            max_articles: Numero massimo articoli (None = tutti)
            auto_save_every: Auto-save ogni N articoli
            verbose: Stampa progress
            
        Returns:
            Dict con statistiche finali
        """
        if verbose:
            print("📚 Starting Wikipedia training")
            print(f"   Source: {self.extractor.dump_path.name}")
            print(f"   Max articles: {max_articles or 'unlimited'}")
            print(f"   Auto-save: every {auto_save_every} articles")
            print()
        
        total_tokens = 0
        articles_processed = 0
        start_time = time.time()
        
        for article_num, (title, text) in enumerate(self.extractor.iter_articles(), 1):
            # Check limit
            if max_articles and article_num > max_articles:
                break
            
            if verbose and article_num % 10 == 0:
                print(f"\n📖 Article {article_num}: {title[:50]}...")
                print(f"   Length: {len(text):,} chars")
            
            # Train on article
            article_start = time.time()
            processed = self.brain.train_on_text(text, passes=1)
            article_time = time.time() - article_start
            
            total_tokens += processed
            articles_processed += 1
            
            # Update stats
            brain_stats = self.brain.get_stats()
            self.stats.update(
                cycles=brain_stats['cycles'],
                tokens=brain_stats['tokens'],
                loss=brain_stats['loss'],
                vocab_size=brain_stats['vocab_words']
            )
            
            if verbose and article_num % 10 == 0:
                tokens_per_sec = processed / article_time if article_time > 0 else 0
                print(f"   Processed: {processed:,} tokens ({tokens_per_sec:.0f} tok/s)")
                print(f"   Loss: {brain_stats['loss']:.4f} {self.stats.get_loss_trend()}")
                print(f"   Vocab: {brain_stats['vocab_words']:,} words")
            
            # Auto-save
            if article_num % auto_save_every == 0:
                ckpt_name = f"wiki_article{article_num}.ckpt"
                ckpt_path = self.checkpoint_manager.get_checkpoint_path(ckpt_name)
                
                if verbose:
                    print(f"   💾 Auto-saving: {ckpt_name}")
                
                self.brain.save_checkpoint(str(ckpt_path))
        
        # Final stats
        elapsed = time.time() - start_time
        final_stats = self.stats.get_summary()
        
        if verbose:
            print("\n" + "=" * 70)
            print("🎉 Wikipedia training complete!")
            print("=" * 70)
            print(f"Articles: {articles_processed}")
            print(f"Tokens: {total_tokens:,}")
            print(f"Time: {elapsed/60:.1f} minutes")
            print(f"Speed: {total_tokens/elapsed:.0f} tokens/sec")
            print(f"Loss: {final_stats['loss']:.4f}")
            print(f"Vocab: {final_stats['vocab_size']:,} words")
            print()
        
        return final_stats


class WikipediaStreamTrainer:
    """Training continuo su Wikipedia via API con gestione RAM"""

    def __init__(self,
                 brain: VectLLMBrain,
                 language: str = 'en',
                 ram_percentage: float = 30.0,
                 stats_collector: Optional[StatsCollector] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None):
        """
        Args:
            brain: Istanza VectLLMBrain
            language: Wikipedia language code
            ram_percentage: Target RAM usage for article buffer
            stats_collector: Collector statistiche
            checkpoint_manager: Manager checkpoint
        """
        self.brain = brain
        self.fetcher = WikipediaAPIFetcher(
            language=language,
            ram_percentage=ram_percentage
        )
        self.stats = stats_collector or StatsCollector()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()

    def train(self,
              max_articles: Optional[int] = None,
              passes_per_batch: int = 1,
              auto_save_every: int = 100,
              verbose: bool = True) -> dict:
        """
        Train continuously on Wikipedia articles.

        Ciclo:
        1. Scarica articoli fino a RAM target
        2. Allena su articoli
        3. Libera memoria
        4. Ripeti

        Args:
            max_articles: Numero massimo articoli totali (None = infinito)
            passes_per_batch: Passate training per ogni batch
            auto_save_every: Auto-save ogni N articoli
            verbose: Stampa progress

        Returns:
            Dict con statistiche finali
        """
        if verbose:
            print("📚 Starting Wikipedia Stream Training")
            print(f"   Language: {self.fetcher.language}")
            print(f"   RAM target: {self.fetcher.ram_percentage}%")
            print(f"   Max articles: {max_articles or 'unlimited'}")
            print(f"   Passes per batch: {passes_per_batch}")
            print()

        total_tokens = 0
        articles_processed = 0
        batch_num = 0
        start_time = time.time()

        try:
            while True:
                # Check if we've reached max articles
                if max_articles and articles_processed >= max_articles:
                    break

                batch_num += 1

                # 1. Fetch batch
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"📦 BATCH {batch_num}")
                    print(f"{'='*60}")

                fetched = self.fetcher.fetch_batch()

                if fetched == 0:
                    print("   ⚠️  No articles fetched, retrying...")
                    time.sleep(5)
                    continue

                # 2. Train on batch
                batch_start = time.time()
                batch_tokens = 0

                for pass_num in range(1, passes_per_batch + 1):
                    if verbose:
                        print(f"\n   📖 Pass {pass_num}/{passes_per_batch}")

                    for i, (title, text) in enumerate(self.fetcher.iter_articles()):
                        article_num = articles_processed + i + 1

                        # Check limit
                        if max_articles and article_num > max_articles:
                            break

                        # Train
                        processed = self.brain.train_on_text(text, passes=1)
                        batch_tokens += processed

                        if pass_num == 1:
                            articles_processed += 1

                        # Progress update
                        if verbose and i % 10 == 0:
                            stats = self.brain.get_stats()
                            print(f"      Article {i+1}/{fetched}: {title[:40]}...")
                            print(f"      Loss: {stats['loss']:.4f}, Vocab: {stats['vocab_words']}")

                        # Auto-save
                        if article_num % auto_save_every == 0:
                            ckpt_name = f"wiki_stream_{article_num}.ckpt"
                            ckpt_path = self.checkpoint_manager.get_checkpoint_path(ckpt_name)

                            if verbose:
                                print(f"      💾 Auto-saving: {ckpt_name}")

                            self.brain.save_checkpoint(str(ckpt_path))

                total_tokens += batch_tokens
                batch_time = time.time() - batch_start

                # Update stats
                brain_stats = self.brain.get_stats()
                self.stats.update(
                    cycles=brain_stats['cycles'],
                    tokens=brain_stats['tokens'],
                    loss=brain_stats['loss'],
                    vocab_size=brain_stats['vocab_words']
                )

                if verbose:
                    tokens_per_sec = batch_tokens / batch_time if batch_time > 0 else 0
                    print(f"\n   ✓ Batch {batch_num} complete")
                    print(f"     Articles: {fetched}")
                    print(f"     Tokens: {batch_tokens:,} ({tokens_per_sec:.0f}/s)")
                    print(f"     Total processed: {articles_processed}")
                    print(f"     Loss: {brain_stats['loss']:.4f}")

                # 3. Clear buffer
                self.fetcher.clear_buffer()

        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")

        # Final stats
        elapsed = time.time() - start_time
        final_stats = self.stats.get_summary()

        if verbose:
            print("\n" + "=" * 70)
            print("🎉 Wikipedia Stream Training complete!")
            print("=" * 70)
            print(f"Batches: {batch_num}")
            print(f"Articles: {articles_processed}")
            print(f"Tokens: {total_tokens:,}")
            print(f"Time: {elapsed/60:.1f} minutes")
            print(f"Speed: {total_tokens/elapsed:.0f} tokens/sec")
            print(f"Loss: {final_stats['loss']:.4f}")
            print(f"Vocab: {final_stats['vocab_size']:,} words")
            print()

        return final_stats


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("=== Wikipedia Training Test ===\n")
    
    # Create test dump (JSONL format)
    test_dump = Path("./test_wiki.jsonl")
    
    articles = [
        {"title": "Test Article 1", "text": "This is test article 1. " * 20},
        {"title": "Test Article 2", "text": "This is test article 2. " * 20},
        {"title": "Test Article 3", "text": "This is test article 3. " * 20},
    ]
    
    with open(test_dump, 'w') as f:
        for article in articles:
            f.write(json.dumps(article) + '\n')
    
    print("Created test dump")
    
    # Test extractor
    extractor = WikipediaExtractor(str(test_dump))
    
    print("\nExtracting articles:")
    for title, text in extractor.iter_articles():
        print(f"  {title}: {len(text)} chars")
    
    # Cleanup
    test_dump.unlink()
    
    print("\n✅ Wikipedia training test complete!")
