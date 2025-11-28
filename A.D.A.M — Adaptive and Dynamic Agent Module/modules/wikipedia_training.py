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
    from core.pipeline import PipelinedTrainer
    from core.config import TRAINING_CONFIG
    from Utils.checkpoint import CheckpointManager
    from modules.training_logger import get_logger
except ImportError:
    from ..core.brain_wrapper import VectLLMBrain
    from ..core.stats import StatsCollector
    from ..core.pipeline import PipelinedTrainer
    from ..core.config import TRAINING_CONFIG
    from ..Utils.checkpoint import CheckpointManager
    from ..modules.training_logger import get_logger


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
                 batch_size: Optional[int] = None,
                 min_article_length: Optional[int] = None,
                 max_article_length: int = 50000):
        """
        Args:
            language: Wikipedia language code (en, it, de, etc.)
            ram_percentage: Target RAM usage percentage for article buffer (legacy, ignored if batch_size > 0)
            batch_size: Number of articles per batch (default: from TRAINING_CONFIG)
            min_article_length: Minimum article length in chars (default: from TRAINING_CONFIG)
            max_article_length: Maximum article length in chars
        """
        # Use config defaults if not specified
        if batch_size is None:
            batch_size = TRAINING_CONFIG.BATCH_SIZE
        if min_article_length is None:
            min_article_length = TRAINING_CONFIG.MIN_TEXT_LENGTH

        self.language = language
        self.ram_percentage = ram_percentage
        self.batch_size = batch_size
        self.min_length = min_article_length
        self.max_length = max_article_length
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"

        # Article buffer
        self.articles: List[Tuple[str, str]] = []
        self.total_fetched = 0
        self.logger = get_logger()

        self.logger.info(f"Wikipedia API Fetcher initialized")
        self.logger.info(f"  Language: {language}")
        self.logger.info(f"  Batch size: {batch_size} articles")

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
            self.logger.warning(f"API error: {e}")
            return {}

    def _get_random_titles(self, count: Optional[int] = None) -> List[str]:
        """Get random article titles"""
        if count is None:
            count = TRAINING_CONFIG.API_BATCH_SIZE
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

    def fetch_batch(self, api_batch_size: Optional[int] = None) -> int:
        """
        Fetch a batch of articles.

        Args:
            api_batch_size: Number of titles to fetch per API call (default: from TRAINING_CONFIG)

        Returns:
            Number of articles fetched in this batch
        """
        if api_batch_size is None:
            api_batch_size = TRAINING_CONFIG.API_BATCH_SIZE
        fetched = 0
        target_articles = self.batch_size

        self.logger.info(f"Fetching {target_articles} articles...")

        while fetched < target_articles:
            titles = self._get_random_titles(api_batch_size)

            if not titles:
                self.logger.warning("No titles received, retrying...")
                time.sleep(1)
                continue

            for title in titles:
                if fetched >= target_articles:
                    break

                content = self._get_article_content(title)

                if content:
                    content = self.clean_text(content)

                    if self.min_length <= len(content) <= self.max_length:
                        self.articles.append((title, content))
                        fetched += 1
                        self.total_fetched += 1

                        if fetched % 10 == 0:
                            self.logger.info(f"  Fetched {fetched}/{target_articles} articles")

            # Small delay to be nice to Wikipedia API
            time.sleep(0.5)

        self.logger.info(f"  Batch complete: {fetched} articles")
        return fetched

    def get_articles(self) -> List[Tuple[str, str]]:
        """Get current article buffer"""
        return self.articles

    def clear_buffer(self):
        """Clear article buffer and free memory"""
        self.articles = []
        gc.collect()
        self.logger.debug(f"Buffer cleared (RAM: {get_memory_usage_percent():.1f}%)")

    def iter_articles(self) -> Iterator[Tuple[str, str]]:
        """Iterate over buffered articles"""
        for title, text in self.articles:
            yield title, text


class WikipediaExtractor:
    """Estrae testo pulito da Wikipedia dump"""

    def __init__(self,
                 dump_path: str,
                 min_article_length: Optional[int] = None,
                 max_article_length: int = 100000):
        """
        Args:
            dump_path: Path al dump Wikipedia (XML o JSONL)
            min_article_length: Lunghezza minima articolo (chars, default: from TRAINING_CONFIG)
            max_article_length: Lunghezza massima articolo (chars)
        """
        if min_article_length is None:
            min_article_length = TRAINING_CONFIG.DATASET_MIN_FILE_LENGTH

        self.dump_path = Path(dump_path)
        self.min_length = min_article_length
        self.max_length = max_article_length
        
        if not self.dump_path.exists():
            raise FileNotFoundError(f"Dump not found: {dump_path}")

        # Detect format
        self.format = self._detect_format()
        logger = get_logger()
        logger.info(f"Wikipedia dump format: {self.format}")
    
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
        self.logger = get_logger()
    
    def train(self,
              max_articles: Optional[int] = None,
              auto_save_every: Optional[int] = None,
              verbose: bool = True) -> dict:
        """
        Train su Wikipedia.

        Args:
            max_articles: Numero massimo articoli (None = tutti)
            auto_save_every: Auto-save ogni N articoli (default: from TRAINING_CONFIG)
            verbose: Stampa progress
            
        Returns:
            Dict con statistiche finali
        """
        if auto_save_every is None:
            auto_save_every = TRAINING_CONFIG.AUTO_SAVE_FREQUENCY

        if verbose:
            self.logger.training_start(
                "Wikipedia Dump",
                source=self.extractor.dump_path.name,
                max_articles=max_articles or 'unlimited',
                auto_save=f"every {auto_save_every} articles"
            )
        
        total_tokens = 0
        articles_processed = 0
        start_time = time.time()
        
        for article_num, (title, text) in enumerate(self.extractor.iter_articles(), 1):
            # Check limit
            if max_articles and article_num > max_articles:
                break
            
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
                self.logger.article_progress(
                    article_num, title, processed, brain_stats['loss'], brain_stats['vocab_words']
                )

            # Auto-save
            if article_num % auto_save_every == 0:
                ckpt_name = f"wiki_article{article_num}.ckpt"
                ckpt_path = self.checkpoint_manager.get_checkpoint_path(ckpt_name)

                if verbose:
                    self.logger.checkpoint_save(ckpt_name)

                self.brain.save_checkpoint(str(ckpt_path))
        
        # Final stats
        elapsed = time.time() - start_time
        final_stats = self.stats.get_summary()
        
        if verbose:
            self.logger.training_end(
                articles=articles_processed,
                tokens=total_tokens,
                speed=total_tokens/elapsed if elapsed > 0 else 0,
                loss=final_stats['loss'],
                vocab=final_stats['vocab_size']
            )
        
        return final_stats


class WikipediaStreamTrainer:
    """Training continuo su Wikipedia via API con gestione RAM"""

    def __init__(self,
                 brain: VectLLMBrain,
                 language: str = 'en',
                 batch_size: Optional[int] = None,
                 stats_collector: Optional[StatsCollector] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None,
                 validation_articles: int = None,
                 validation_frequency: int = None,
                 early_stopping_patience: int = None,
                 validate_per_pass: bool = True):
        """
        Args:
            brain: Istanza VectLLMBrain
            language: Wikipedia language code
            batch_size: Number of articles per batch (default: from TRAINING_CONFIG)
            stats_collector: Collector statistiche
            checkpoint_manager: Manager checkpoint
            validation_articles: Number of articles to use for validation (default: 10% of batch_size)
            validation_frequency: Validate every N articles (only if validate_per_pass=False)
            early_stopping_patience: Stop after N validations without improvement (default from config)
            validate_per_pass: If True, validate only at end of each pass; if False, validate every N articles
        """
        if batch_size is None:
            batch_size = TRAINING_CONFIG.BATCH_SIZE

        self.brain = brain
        self.fetcher = WikipediaAPIFetcher(
            language=language,
            batch_size=batch_size
        )
        self.stats = stats_collector or StatsCollector()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.logger = get_logger()

        # Validation settings
        self.validation_articles = validation_articles or max(1, int(batch_size * 0.1))
        self.validation_frequency = validation_frequency or TRAINING_CONFIG.VALIDATION_FREQUENCY
        self.early_stopping_patience = early_stopping_patience or TRAINING_CONFIG.EARLY_STOPPING_PATIENCE
        self.validate_per_pass = validate_per_pass

        # Validation state
        self.val_articles: List[Tuple[str, str]] = []
        self.best_val_loss = float('inf')
        self.validations_without_improvement = 0

    def _fetch_validation_articles(self):
        """Fetch articles for validation set."""
        if self.validation_articles <= 0:
            return

        self.logger.info(f"Fetching {self.validation_articles} validation articles...")

        # Create temporary fetcher for validation
        val_fetcher = WikipediaAPIFetcher(
            language=self.fetcher.language,
            batch_size=self.validation_articles
        )
        val_fetcher.fetch_batch()
        self.val_articles = list(val_fetcher.get_articles())

        self.logger.info(f"  Validation set: {len(self.val_articles)} articles")

    def _run_validation(self) -> float:
        """
        Run validation on validation articles.

        Returns:
            Average validation loss
        """
        if not self.val_articles:
            return -1.0

        # Defer GPU syncs during validation to avoid contention
        self.brain.begin_validation()

        total_loss = 0.0
        count = 0

        for title, text in self.val_articles:
            loss = self.brain.validate_on_text(text)
            if loss >= 0:
                total_loss += loss
                count += 1

        # Resume normal sync after validation
        self.brain.end_validation()

        if count == 0:
            return -1.0

        avg_loss = total_loss / count

        # Update stats
        self.stats.metrics.validation_loss = avg_loss
        self.stats.metrics.validation_history.append((time.time(), avg_loss))

        # Check for improvement
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.stats.metrics.best_validation_loss = avg_loss
            self.validations_without_improvement = 0
        else:
            self.validations_without_improvement += 1
            self.stats.metrics.validations_without_improvement = self.validations_without_improvement

        return avg_loss

    def train(self,
              max_articles: Optional[int] = None,
              passes_per_batch: int = 1,
              auto_save_every: Optional[int] = None,
              verbose: bool = True,
              use_pipeline: bool = True,
              prefetch_size: int = 3,
              enable_validation: bool = True,
              enable_early_stopping: bool = True) -> dict:
        """
        Train continuously on Wikipedia articles.

        Ciclo:
        1. Scarica articoli fino a RAM target
        2. Allena su articoli (pipelined per overlap CPU/GPU)
        3. Libera memoria
        4. Ripeti

        Args:
            max_articles: Numero massimo articoli totali (None = infinito)
            passes_per_batch: Passate training per ogni batch
            auto_save_every: Auto-save ogni N articoli (default: from TRAINING_CONFIG)
            verbose: Stampa progress
            use_pipeline: DEPRECATED - pipelined training is now always used
            prefetch_size: Numero batch da pre-caricare
            enable_validation: Run validation during training
            enable_early_stopping: Stop if validation doesn't improve

        Returns:
            Dict con statistiche finali
        """
        if auto_save_every is None:
            auto_save_every = TRAINING_CONFIG.AUTO_SAVE_FREQUENCY

        # Fetch validation articles first
        if enable_validation and not self.val_articles:
            self._fetch_validation_articles()

        if verbose:
            self.logger.training_start(
                "Wikipedia Stream",
                language=self.fetcher.language,
                batch_size=f"{self.fetcher.batch_size} articles",
                validation_articles=len(self.val_articles) if enable_validation else 0,
                max_articles=max_articles or 'unlimited',
                passes_per_batch=passes_per_batch,
                pipeline='enabled' if use_pipeline else 'disabled',
                validation='enabled' if enable_validation and self.val_articles else 'disabled'
            )

        total_tokens = 0
        articles_processed = 0
        batch_num = 0
        start_time = time.time()
        early_stopped = False

        try:
            while True:
                # Check if we've reached max articles
                if max_articles and articles_processed >= max_articles:
                    break

                # Check early stopping
                if enable_early_stopping and self.validations_without_improvement >= self.early_stopping_patience:
                    if verbose:
                        self.logger.validation_early_stop(self.early_stopping_patience)
                    early_stopped = True
                    break

                batch_num += 1

                # 1. Fetch batch
                if verbose:
                    self.logger.info(f"BATCH {batch_num}")

                fetched = self.fetcher.fetch_batch()

                if fetched == 0:
                    self.logger.warning("No articles fetched, retrying...")
                    time.sleep(5)
                    continue

                # 2. Train on batch
                batch_start = time.time()
                batch_tokens = 0

                for pass_num in range(1, passes_per_batch + 1):
                    if verbose:
                        self.logger.pass_start(pass_num, passes_per_batch)

                    # Pipelined training - overlap CPU encoding with GPU training
                    batch_tokens += self._train_batch_pipelined(
                        fetched, pass_num, max_articles, articles_processed,
                        auto_save_every, verbose, prefetch_size,
                        enable_validation, enable_early_stopping
                    )
                    if pass_num == 1:
                        articles_processed += fetched
                        if max_articles:
                            articles_processed = min(articles_processed, max_articles)

                total_tokens += batch_tokens
                batch_time = time.time() - batch_start

                # Get latest stats for display
                brain_stats = self.brain.get_stats()

                if verbose:
                    tokens_per_sec = batch_tokens / batch_time if batch_time > 0 else 0
                    self.logger.batch_progress(batch_num, batch_tokens, brain_stats['loss'], tokens_per_sec)

                # 3. Clear buffer
                self.fetcher.clear_buffer()

        except KeyboardInterrupt:
            self.logger.interrupted()

        # Final stats
        elapsed = time.time() - start_time
        final_stats = self.stats.get_summary()

        # Add validation stats
        final_stats['validation_loss'] = self.stats.metrics.validation_loss
        final_stats['best_validation_loss'] = self.best_val_loss
        final_stats['early_stopped'] = early_stopped

        if verbose:
            end_kwargs = {
                'batches': batch_num,
                'articles': articles_processed,
                'tokens': total_tokens,
                'speed': total_tokens/elapsed if elapsed > 0 else 0,
                'loss': final_stats['loss'],
                'vocab': final_stats['vocab_size']
            }
            if self.val_articles:
                end_kwargs['validation_loss'] = self.stats.metrics.validation_loss
                end_kwargs['best_val_loss'] = self.best_val_loss
            if early_stopped:
                end_kwargs['status'] = 'early_stopped'

            self.logger.training_end(**end_kwargs)

        return final_stats

    def _train_batch_pipelined(self,
                                fetched: int,
                                pass_num: int,
                                max_articles: Optional[int],
                                articles_processed: int,
                                auto_save_every: int,
                                verbose: bool,
                                prefetch_size: int,
                                enable_validation: bool = True,
                                enable_early_stopping: bool = True) -> int:
        """
        Train on batch using pipelined CPU/GPU overlap.

        Args:
            fetched: Number of articles fetched
            pass_num: Current pass number
            max_articles: Max articles limit
            articles_processed: Articles processed so far
            auto_save_every: Auto-save interval
            verbose: Verbose output
            prefetch_size: Prefetch queue size
            enable_validation: Run validation during training
            enable_early_stopping: Stop if validation doesn't improve

        Returns:
            Tokens processed in this batch
        """
        # Begin deferred sync - batch all vocab syncs until end of pass
        self.brain.begin_deferred_sync()

        # Create pipelined trainer
        trainer = PipelinedTrainer(self.brain, prefetch_size=prefetch_size)

        # Collect texts from fetcher
        texts = []
        article_info = []  # Store (title, article_num) for progress tracking

        for i, (title, text) in enumerate(self.fetcher.iter_articles()):
            article_num = articles_processed + i + 1

            # Check limit
            if max_articles and article_num > max_articles:
                break

            texts.append(text)
            article_info.append((title, article_num))

        if not texts:
            return 0

        # Progress callback
        processed_count = [0]  # Use list for closure
        last_save_article = [articles_processed]
        total_batches = len(texts)  # Approximate batches (actual may vary slightly)

        def progress_callback(batch_num: int, tokens: int):
            processed_count[0] += 1

            # Map batch to article (handle case where batches > articles)
            if processed_count[0] <= len(article_info):
                idx = processed_count[0] - 1
                title, article_num = article_info[idx]

                # Progress update (only show for actual articles, not internal chunks)
                if verbose and processed_count[0] % 10 == 0:
                    stats = self.brain.get_stats()
                    self.logger.article_progress(
                        processed_count[0], title, tokens, stats['loss'], stats['vocab_words']
                    )
            else:
                # More batches than articles - pipeline is chunking internally
                # Use last article info for validation/save checks
                idx = len(article_info) - 1
                _, article_num = article_info[idx]
                title = f"(internal chunk {processed_count[0] - len(article_info)})"
                # Don't log these - they're just internal pipeline chunks

            # Validation check (only for actual articles, not internal chunks)
            if (processed_count[0] <= len(article_info) and
                enable_validation and self.val_articles and
                not self.validate_per_pass and
                article_num % self.validation_frequency == 0):
                if verbose:
                    self.logger.validation_start(len(self.val_articles))
                val_loss = self._run_validation()
                if verbose:
                    improved = (self.validations_without_improvement == 0)
                    self.logger.validation_result(val_loss, self.best_val_loss, improved)
                    self.logger.validation_complete()

            # Auto-save check (only for actual articles, not internal chunks)
            if (processed_count[0] <= len(article_info) and
                article_num % auto_save_every == 0 and
                article_num > last_save_article[0]):
                last_save_article[0] = article_num
                ckpt_name = f"wiki_stream_{article_num}.ckpt"
                ckpt_path = self.checkpoint_manager.get_checkpoint_path(ckpt_name)

                if verbose:
                    self.logger.checkpoint_save(ckpt_name)

                self.brain.save_checkpoint(str(ckpt_path))

            # Update stats
            brain_stats = self.brain.get_stats()
            self.stats.update(
                cycles=brain_stats['cycles'],
                tokens=brain_stats['tokens'],
                loss=brain_stats['loss'],
                vocab_size=brain_stats['vocab_words']
            )

        # Run pipelined training
        batch_tokens = trainer.train_texts(iter(texts), progress_callback)

        # Pass completion message
        if verbose:
            stats = self.brain.get_stats()
            self.logger.info("=" * 70)
            self.logger.info(f"✅ PASS {pass_num} COMPLETE - Loss: {stats['loss']:.4f}, Vocab: {stats['vocab_words']} words")
            self.logger.info("=" * 70)

        # Validate at end of pass (if validate_per_pass is True)
        if enable_validation and self.val_articles and self.validate_per_pass:
            if verbose:
                self.logger.validation_start(len(self.val_articles))
            val_loss = self._run_validation()
            if verbose:
                improved = (self.validations_without_improvement == 0)
                self.logger.validation_result(val_loss, self.best_val_loss, improved)
                self.logger.validation_complete()

        # End deferred sync - execute all batched vocab syncs
        self.brain.end_deferred_sync()

        # Get pipeline stats
        pipeline_stats = trainer.get_stats()
        if verbose and pass_num == 1:
            self.logger.pipeline_stats(
                pipeline_stats['throughput_tok_s'],
                pipeline_stats.get('gpu_utilization', 0)
            )

        return batch_tokens


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
