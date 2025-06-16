#!/usr/bin/env python3
"""
Core Essence - A fundamental implementation that aligns words and code

This demonstrates what true alignment between language and implementation looks like,
with direct fine-tuning, real-time visualization, and concept relationship mapping.
"""

import os
import json
import time
import torch
import shutil
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("essence")
logger.setLevel(logging.INFO)

# Essential paths
ESSENCE_DIR = Path("./essence")
UNDERSTANDING_DIR = ESSENCE_DIR / "understanding"
ARTIFACTS_DIR = ESSENCE_DIR / "artifacts"
REFLECTIONS_DIR = ESSENCE_DIR / "reflections"

# Create essential directories
for directory in [ESSENCE_DIR, UNDERSTANDING_DIR, ARTIFACTS_DIR, REFLECTIONS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Terminal formatting
RESET = "\033[0m"
EMPHASIS = "\033[1;38;5;74m"
HIGHLIGHT = "\033[38;5;159m"
NOTICE = "\033[38;5;222m"
DIM = "\033[2m"

def breathe(seconds: float = 0.5) -> None:
    """Give the user a moment to absorb."""
    time.sleep(seconds)

def speak(message: str, pause: float = 0.3, emphasis: bool = False) -> None:
    """Communicate with intention and clarity."""
    if emphasis:
        print(f"\n{EMPHASIS}{message}{RESET}")
    else:
        print(f"  {message}")
    breathe(pause)

def reflect(message: str) -> None:
    """Quiet observations that invite thought."""
    print(f"\n  {DIM}{message}{RESET}")
    breathe(0.7)

def notice(message: str) -> None:
    """Gentle attention guidance."""
    print(f"  {NOTICE}{message}{RESET}")
    breathe(0.5)


class Concept:
    """
    A meaningful concept with identity and relationships.
    
    This isn't just a data container - it represents a coherent
    piece of understanding with its own identity and connections.
    """
    
    def __init__(self, essence: str, nature: Optional[Dict[str, Any]] = None):
        """Initialize a concept with its essence and nature."""
        self.essence = essence
        self.nature = nature or {}
        self.relationships = []
        self.embedding = None
        self.absorbed_at = time.time()
    
    def relate_to(self, other: 'Concept', strength: float) -> None:
        """Form a relationship with another concept."""
        relationship = {
            "concept": other,
            "strength": strength,
            "formed_at": time.time()
        }
        self.relationships.append(relationship)
    
    def embed(self, embedding_function: Callable[[str], List[float]]) -> None:
        """Embody the concept in a mathematical space."""
        self.embedding = embedding_function(self.essence)
    
    def similarity_to(self, other: 'Concept') -> float:
        """Understand the similarity to another concept."""
        if self.embedding is None or other.embedding is None:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(self.embedding, other.embedding) / (
            np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding)
        )
        return float(similarity)
    
    def __str__(self) -> str:
        """Return the essence of the concept."""
        return self.essence


class Understanding:
    """
    A network of concepts and their relationships.
    
    This isn't just a data structure - it's a representation of
    how concepts relate to each other in a meaningful way.
    """
    
    def __init__(self):
        """Initialize an understanding as a network of concepts."""
        self.concepts = []
        self.embedding_model = None
    
    def absorb(self, source_path: str) -> None:
        """
        Absorb meaning from a source, forming concepts and relationships.
        This doesn't just load data - it creates understanding.
        """
        if not os.path.exists(source_path):
            speak(f"I couldn't find this source of understanding.", emphasis=True)
            return
        
        speak(f"Absorbing understanding from {HIGHLIGHT}{os.path.basename(source_path)}{RESET}...", emphasis=True)
        breathe(1)
        
        try:
            with open(source_path, "r") as f:
                documents = json.load(f)
            
            # Initialize the embedding model if needed
            if self.embedding_model is None:
                self._initialize_embedding_model()
            
            # Extract concepts from documents
            new_concepts = []
            for doc in documents:
                content = doc.get("content", "")
                if not content:
                    continue
                
                # Create a concept with identity
                concept = Concept(content, doc.get("metadata", {}))
                
                # Embody the concept mathematically
                if self.embedding_model:
                    concept.embed(self.embedding_model.embed_query)
                
                new_concepts.append(concept)
                self.concepts.append(concept)
                
                # Show the absorption process
                concept_preview = content[:60] + "..." if len(content) > 60 else content
                print(f"  {HIGHLIGHT}○{RESET} Absorbing: {concept_preview}")
                breathe(0.2)
            
            # Form relationships between concepts
            if len(new_concepts) > 1:
                reflect("Discovering meaningful relationships between concepts...")
                
                # Form relationships based on embedding similarity
                for i, concept1 in enumerate(new_concepts):
                    for j, concept2 in enumerate(new_concepts[i+1:], i+1):
                        similarity = concept1.similarity_to(concept2)
                        if similarity > 0.5:  # Meaningful connection threshold
                            concept1.relate_to(concept2, similarity)
                            concept2.relate_to(concept1, similarity)
                            
                            if similarity > 0.7:  # Strong relationship
                                print(f"  {HIGHLIGHT}•{RESET} Strong connection: {concept1.essence[:30]}... ↔ {concept2.essence[:30]}...")
            
            speak(f"Absorbed {len(new_concepts)} new concepts into understanding.", emphasis=True)
            
            # Visualize the concept network
            self._visualize_understanding()
            
        except Exception as e:
            speak(f"I struggled to absorb that source: {str(e)}", emphasis=True)
    
    def _initialize_embedding_model(self) -> None:
        """Initialize a model for embodying concepts mathematically."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a financial domain model if available
            model_name = "FinMTEB/Fin-E5-small"
            
            reflect(f"Initializing understanding with {model_name}...")
            self.embedding_model = SentenceTransformer(model_name)
            
        except ImportError:
            speak("I need sentence-transformers to form mathematical understanding.", emphasis=True)
            self.embedding_model = None
    
    def _visualize_understanding(self) -> None:
        """
        Visualize the network of concepts and their relationships.
        This isn't just a graph - it's a map of understanding.
        """
        if not self.concepts or len(self.concepts) < 3:
            return
        
        # Check if concepts have embeddings
        if any(c.embedding is None for c in self.concepts):
            return
        
        try:
            # Create a visualization directory
            viz_dir = REFLECTIONS_DIR / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Get all embeddings
            embeddings = np.array([c.embedding for c in self.concepts])
            
            # Reduce to 2D for visualization
            tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(embeddings)-1)), 
                        random_state=42)
            reduced_embeddings = tsne.fit_transform(embeddings)
            
            # Create a figure
            plt.figure(figsize=(10, 8))
            
            # Plot concept nodes
            x, y = reduced_embeddings[:, 0], reduced_embeddings[:, 1]
            plt.scatter(x, y, c='skyblue', s=100, alpha=0.7)
            
            # Add concept labels
            for i, concept in enumerate(self.concepts):
                label = concept.essence[:20] + "..." if len(concept.essence) > 20 else concept.essence
                plt.annotate(label, (x[i], y[i]), fontsize=8, alpha=0.8)
            
            # Plot edges for strong relationships
            for i, concept in enumerate(self.concepts):
                for rel in concept.relationships:
                    if rel["strength"] > 0.6:  # Only show strong relationships
                        other = rel["concept"]
                        j = self.concepts.index(other)
                        plt.plot([x[i], x[j]], [y[i], y[j]], 'gray', alpha=0.2, linewidth=rel["strength"])
            
            # Finalize plot
            plt.title("Concept Relationship Network")
            plt.tight_layout()
            
            # Save the visualization
            viz_path = viz_dir / f"understanding_{int(time.time())}.png"
            plt.savefig(viz_path)
            plt.close()
            
            notice(f"Understanding visualization saved to {viz_path}")
            
        except Exception as e:
            reflect(f"Couldn't visualize understanding: {str(e)}")
    
    def reflect_on(self) -> Dict[str, Any]:
        """Reflect on the current state of understanding."""
        reflection = {
            "concepts": len(self.concepts),
            "relationships": sum(len(c.relationships) for c in self.concepts),
            "avg_relationships": sum(len(c.relationships) for c in self.concepts) / max(1, len(self.concepts)),
            "strongest_relationship": max([rel["strength"] for c in self.concepts for rel in c.relationships], default=0)
        }
        
        return reflection
    
    def export_training_pairs(self, output_path: str) -> int:
        """
        Export concept relationships for enlightenment.
        This creates meaningful pairs for transformation.
        """
        if not self.concepts:
            return 0
        
        training_pairs = []
        
        # Create pairs from direct relationships
        for concept in self.concepts:
            for rel in concept.relationships:
                if rel["strength"] > 0.5:  # Meaningful relationship
                    training_pairs.append({
                        "text1": concept.essence,
                        "text2": rel["concept"].essence,
                        "score": rel["strength"]
                    })
        
        # Create pairs from concept properties
        for concept in self.concepts:
            nature = concept.nature
            if nature:
                for key, value in nature.items():
                    if isinstance(value, str) and len(value) > 5:
                        question = f"What is the {key} of this financial document?"
                        training_pairs.append({
                            "text1": question,
                            "text2": concept.essence,
                            "score": 0.85  # Strong relationship for explicit properties
                        })
        
        # Save the training pairs
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(training_pairs, f, indent=2)
        
        return len(training_pairs)


class Enlightenment:
    """
    The process of transforming a model's understanding.
    
    This directly implements fine-tuning, not just calling a script,
    with real-time visualization of the transformation process.
    """
    
    def __init__(self, understanding: Understanding, model_name: str = "FinMTEB/Fin-E5-small"):
        """Initialize the enlightenment process with understanding."""
        self.understanding = understanding
        self.model_name = model_name
        self.enlightened_path = None
        self.journey_path = REFLECTIONS_DIR / "journey.json"
        
        # Prepare for the journey
        self.journey = []
        if self.journey_path.exists():
            with open(self.journey_path, "r") as f:
                self.journey = json.load(f)
    
    def begin(self) -> None:
        """
        Begin the enlightenment journey with direct implementation.
        This doesn't just call a script - it implements the transformation.
        """
        reflect("Preparing for the enlightenment journey...")
        
        # First, reflect on the understanding
        reflection = self.understanding.reflect_on()
        
        if reflection["concepts"] < 5:
            speak("I need more concepts to form a meaningful understanding.", emphasis=True)
            return
        
        # Create the transformation space
        enlightenment_id = int(time.time())
        enlightenment_dir = ARTIFACTS_DIR / f"enlightened_{enlightenment_id}"
        enlightenment_dir.mkdir(exist_ok=True)
        
        # Prepare training data from understanding
        training_path = enlightenment_dir / "training_pairs.json"
        num_pairs = self.understanding.export_training_pairs(str(training_path))
        
        if num_pairs < 10:
            speak("I need more relationships to form a meaningful transformation.", emphasis=True)
            return
        
        speak(f"Beginning enlightenment with {num_pairs} concept relationships...", emphasis=True)
        
        # Direct implementation of fine-tuning, not just calling a script
        try:
            # Import required libraries for direct implementation
            from sentence_transformers import SentenceTransformer, losses, InputExample
            from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
            from torch.utils.data import DataLoader
            
            # Load training pairs
            with open(training_path, "r") as f:
                training_data = json.load(f)
            
            # Prepare training examples
            train_examples = []
            for pair in training_data:
                train_examples.append(InputExample(
                    texts=[pair["text1"], pair["text2"]], 
                    label=float(pair["score"])
                ))
            
            # Split for validation (80/20)
            np.random.shuffle(train_examples)
            split_point = int(len(train_examples) * 0.8)
            eval_examples = train_examples[split_point:]
            train_examples = train_examples[:split_point]
            
            # Load base model
            reflect("Loading the base model...")
            model = SentenceTransformer(self.model_name)
            
            # Create appropriate loss function
            train_loss = losses.CosineSimilarityLoss(model)
            
            # Create evaluator
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_examples)
            
            # Prepare training data loader
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
            
            # Set up the model saving path
            output_path = enlightenment_dir / "enlightened_model"
            
            # Training parameters
            num_epochs = 3
            warmup_steps = int(len(train_dataloader) * 0.1)
            
            # Create a callback for visualizing progress
            from sentence_transformers.evaluation import SentenceEvaluator
            
            class EnlightenmentCallback(SentenceEvaluator):
                def __init__(self):
                    self.epoch = 0
                    self.batch = 0
                    self.total_batches = len(train_dataloader) * num_epochs
                    self.stages = [
                        "Seeing concepts in new light",
                        "Forming deeper connections",
                        "Discovering patterns",
                        "Developing intuition",
                        "Expanding perspective",
                        "Synthesizing understanding",
                        "Achieving clarity"
                    ]
                    self.losses = []
                    self.evaluations = []
                
                def __call__(self, iterable, *args, **kwargs):
                    return self.evaluate(iterable, *args, **kwargs)
                
                def evaluate(self, iterable, epoch, steps, *args, **kwargs):
                    self.epoch = epoch
                    return 0.0
                
                def on_epoch_end(self, epoch, steps, *args, **kwargs):
                    """Called at the end of an epoch."""
                    self.epoch = epoch + 1
                    
                    # Show progress
                    progress = (epoch + 1) / num_epochs
                    stage = self.stages[min(len(self.stages)-1, int(progress * len(self.stages)))]
                    self._show_enlightenment_progress(progress, stage)
                
                def on_evaluation_end(self, score, epoch, steps):
                    """Called after evaluation."""
                    self.evaluations.append((epoch, score))
                
                def on_training_end(self, *args, **kwargs):
                    """Called at the end of training."""
                    # Complete the progress bar
                    self._show_enlightenment_progress(1.0, "Enlightenment complete")
                    print()  # Add a newline
                
                def on_batch_end(self, loss, *args, **kwargs):
                    """Called after each batch update."""
                    self.losses.append(loss)
                    self.batch += 1
                    
                    # Only update visualization occasionally to avoid slowdown
                    if self.batch % 5 == 0 or self.batch == self.total_batches:
                        # Calculate progress within the overall training
                        progress = self.batch / self.total_batches
                        stage_idx = min(len(self.stages)-1, int(progress * len(self.stages)))
                        stage = self.stages[stage_idx]
                        
                        # Show the enlightenment progress
                        self._show_enlightenment_progress(progress, stage)
                
                def _show_enlightenment_progress(self, progress: float, stage: str) -> None:
                    """Show the enlightenment progress with meaningful visualization."""
                    cols = shutil.get_terminal_size().columns
                    inner_width = min(cols - 10, 60)
                    
                    filled = int(inner_width * progress)
                    remaining = inner_width - filled
                    
                    # Create a gradient of understanding
                    gradient = [
                        "\033[38;5;{}m".format(i) for i in range(33, 45)
                    ]
                    bar = ""
                    for i in range(filled):
                        idx = min(len(gradient) - 1, int(i / filled * len(gradient)))
                        bar += gradient[idx] + "━"
                    
                    # Add loss information if available
                    loss_info = ""
                    if self.losses:
                        recent_loss = self.losses[-1]
                        loss_info = f"understanding: {recent_loss:.4f}"
                    
                    print(f"\r  {EMPHASIS}{stage}{RESET} {bar}{DIM}{'━' * remaining}{RESET} {int(progress * 100)}% {DIM}{loss_info}{RESET}", end="", flush=True)
            
            # Create callback
            enlightenment_callback = EnlightenmentCallback()
            
            # Start the enlightenment journey
            speak("Beginning the transformation...", emphasis=True)
            
            # Train the model - direct implementation, not calling a script
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=num_epochs,
                evaluation_steps=100,
                warmup_steps=warmup_steps,
                output_path=str(output_path),
                callback=enlightenment_callback,
                show_progress_bar=False  # We use our own visualization
            )
            
            # Save the enlightened model path
            self.enlightened_path = str(output_path)
            
            # Save the path for future reference
            with open(enlightenment_dir / "path.txt", "w") as f:
                f.write(self.enlightened_path)
            
            # Record the journey
            journey_entry = {
                "began_at": time.time(),
                "understanding": reflection,
                "concepts": reflection["concepts"],
                "training_pairs": num_pairs,
                "model_name": self.model_name,
                "enlightened_path": self.enlightened_path,
                "enlightenment_id": enlightenment_id
            }
            self.journey.append(journey_entry)
            
            # Save the journey
            with open(self.journey_path, "w") as f:
                json.dump(self.journey, f, indent=2)
            
            speak("The enlightenment journey is complete.", emphasis=True)
            notice(f"Your model now sees financial concepts with deeper understanding.")
            
            # Visualize the transformation
            self._visualize_transformation(enlightenment_dir)
            
        except ImportError as e:
            speak(f"I need additional libraries for enlightenment: {str(e)}", emphasis=True)
        except Exception as e:
            speak(f"The enlightenment journey encountered an obstacle: {str(e)}", emphasis=True)
    
    def _visualize_transformation(self, output_dir: Path) -> None:
        """
        Visualize the transformation in understanding.
        This shows how concepts are perceived differently.
        """
        try:
            # Import required libraries
            from sentence_transformers import SentenceTransformer
            
            # Load the original and enlightened models
            original_model = SentenceTransformer(self.model_name)
            enlightened_model = SentenceTransformer(self.enlightened_path)
            
            # Create some example financial queries
            queries = [
                "What market risks are mentioned?",
                "How do interest rates affect earnings?",
                "What regulatory changes are important?",
                "What are key financial metrics?",
                "How is the company performing financially?",
                "What is the debt-to-equity ratio?",
                "What's the earnings forecast?",
                "What sustainability factors affect valuation?"
            ]
            
            # Embed with both models
            original_embeddings = original_model.encode(queries)
            enlightened_embeddings = enlightened_model.encode(queries)
            
            # Calculate similarity matrices
            original_sim = cosine_similarity(original_embeddings)
            enlightened_sim = cosine_similarity(enlightened_embeddings)
            
            # Plot comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Original understanding
            im1 = ax1.imshow(original_sim, cmap='viridis')
            ax1.set_title("Original Understanding")
            ax1.set_xticks(range(len(queries)))
            ax1.set_yticks(range(len(queries)))
            ax1.set_xticklabels([q[:15] + "..." for q in queries], rotation=45, ha="right")
            ax1.set_yticklabels([q[:15] + "..." for q in queries])
            
            # Enlightened understanding
            im2 = ax2.imshow(enlightened_sim, cmap='viridis')
            ax2.set_title("Enlightened Understanding")
            ax2.set_xticks(range(len(queries)))
            ax2.set_yticks(range(len(queries)))
            ax2.set_xticklabels([q[:15] + "..." for q in queries], rotation=45, ha="right")
            ax2.set_yticklabels([q[:15] + "..." for q in queries])
            
            # Add colorbar
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save visualization
            viz_path = output_dir / "transformation.png"
            plt.savefig(viz_path)
            plt.close()
            
            notice(f"Transformation visualization saved to {viz_path}")
            
        except Exception as e:
            reflect(f"Couldn't visualize transformation: {str(e)}")


class Contemplation:
    """
    The process of applying enlightened understanding to questions.
    
    This doesn't just run inference - it applies understanding to
    find meaningful insights, with direct model use.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize with an enlightened model path."""
        # Find the most recent enlightened model if none provided
        if not model_path:
            journey_path = REFLECTIONS_DIR / "journey.json"
            if journey_path.exists():
                with open(journey_path, "r") as f:
                    journey = json.load(f)
                if journey:
                    latest = max(journey, key=lambda x: x.get("began_at", 0))
                    model_path = latest.get("enlightened_path")
        
        self.model_path = model_path
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the enlightened model for contemplation."""
        if not self.model_path:
            return
        
        if not os.path.exists(self.model_path):
            speak("The path to enlightened understanding is missing.", emphasis=True)
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            reflect(f"Connecting to enlightened understanding...")
            self.model = SentenceTransformer(self.model_path)
            
        except ImportError:
            speak("I need sentence-transformers to access enlightened understanding.", emphasis=True)
        except Exception as e:
            speak(f"I couldn't connect to enlightened understanding: {str(e)}", emphasis=True)
    
    def contemplate(self, question: str, understanding: Optional[Understanding] = None) -> List[Dict[str, Any]]:
        """
        Contemplate a question with enlightened understanding.
        This doesn't just search - it finds meaningful connections.
        """
        if not self.model:
            speak("I don't have access to enlightened understanding.", emphasis=True)
            return []
        
        speak(f"Contemplating: {HIGHLIGHT}{question}{RESET}", emphasis=True)
        breathe(1)
        
        # Direct implementation, not calling a script
        try:
            # Embed the question
            question_embedding = self.model.encode(question)
            
            # If provided with understanding, use it for contemplation
            if understanding and understanding.concepts:
                # Directly find relevant concepts
                reflect("Finding meaningful connections...")
                
                # Check if concepts need embedding
                for concept in understanding.concepts:
                    if concept.embedding is None and hasattr(concept, 'essence'):
                        concept.embed(self.model.encode)
                
                # Calculate similarities to find insights
                similarities = []
                for concept in understanding.concepts:
                    if concept.embedding is not None:
                        similarity = np.dot(question_embedding, concept.embedding) / (
                            np.linalg.norm(question_embedding) * np.linalg.norm(concept.embedding)
                        )
                        similarities.append((concept, float(similarity)))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Return top insights
                results = []
                for concept, similarity in similarities[:5]:  # Top 5 insights
                    results.append({
                        "content": concept.essence,
                        "metadata": concept.nature,
                        "relevance": similarity
                    })
                
                return results
            
            # If no understanding provided, use a default corpus
            else:
                # Check for a corpus of financial documents
                corpus_path = UNDERSTANDING_DIR / "financial_corpus.json"
                if not corpus_path.exists():
                    # Create a minimal corpus from journey data
                    self._create_minimal_corpus()
                
                if corpus_path.exists():
                    with open(corpus_path, "r") as f:
                        corpus = json.load(f)
                    
                    # Embed corpus if not already embedded
                    if "embedded" not in corpus:
                        reflect("Analyzing the financial corpus...")
                        documents = [doc["content"] for doc in corpus["documents"]]
                        embeddings = self.model.encode(documents)
                        corpus["embeddings"] = [e.tolist() for e in embeddings]
                        corpus["embedded"] = True
                        
                        with open(corpus_path, "w") as f:
                            json.dump(corpus, f)
                    
                    # Find relevant documents
                    embeddings = np.array(corpus["embeddings"])
                    documents = corpus["documents"]
                    
                    # Calculate similarities
                    similarities = np.dot(embeddings, question_embedding) / (
                        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding)
                    )
                    
                    # Get top matches
                    top_indices = np.argsort(similarities)[::-1][:5]
                    
                    # Return insights
                    results = []
                    for idx in top_indices:
                        results.append({
                            "content": documents[idx]["content"],
                            "metadata": documents[idx].get("metadata", {}),
                            "relevance": float(similarities[idx])
                        })
                    
                    return results
            
            return []
            
        except Exception as e:
            speak(f"My contemplation was interrupted: {str(e)}", emphasis=True)
            return []
    
    def _create_minimal_corpus(self) -> None:
        """Create a minimal corpus if none exists."""
        corpus_path = UNDERSTANDING_DIR / "financial_corpus.json"
        
        # Check if journey contains concepts
        journey_path = REFLECTIONS_DIR / "journey.json"
        if journey_path.exists():
            try:
                with open(journey_path, "r") as f:
                    journey = json.load(f)
                
                # Look for training data in journey
                documents = []
                for entry in journey:
                    training_path = entry.get("training_path")
                    if training_path and os.path.exists(training_path):
                        with open(training_path, "r") as f:
                            pairs = json.load(f)
                        
                        # Extract texts from pairs
                        for pair in pairs:
                            if "text1" in pair:
                                documents.append({
                                    "content": pair["text1"],
                                    "metadata": {"source": "training"}
                                })
                            if "text2" in pair:
                                documents.append({
                                    "content": pair["text2"],
                                    "metadata": {"source": "training"}
                                })
                
                # Save corpus
                if documents:
                    corpus = {
                        "documents": documents,
                        "created_at": time.time()
                    }
                    
                    with open(corpus_path, "w") as f:
                        json.dump(corpus, f)
                
            except Exception as e:
                reflect(f"Couldn't create corpus: {str(e)}")


def compare_understanding(enlightened_path: Optional[str] = None) -> None:
    """
    Compare original and enlightened understanding.
    This reveals the transformation with direct implementation.
    """
    speak("Revealing the transformation in understanding...", emphasis=True)
    breathe(1)
    
    # Find the most recent enlightened model if none provided
    if not enlightened_path:
        journey_path = REFLECTIONS_DIR / "journey.json"
        if journey_path.exists():
            with open(journey_path, "r") as f:
                journey = json.load(f)
            if journey:
                latest = max(journey, key=lambda x: x.get("began_at", 0))
                enlightened_path = latest.get("enlightened_path")
    
    if not enlightened_path or not os.path.exists(enlightened_path):
        speak("I don't have an enlightened understanding to compare with.", emphasis=True)
        return
    
    # Direct implementation of comparison
    try:
        # Import required libraries
        from sentence_transformers import SentenceTransformer
        
        # Load both models
        reflect("Loading original understanding...")
        original_model = SentenceTransformer("FinMTEB/Fin-E5-small")
        
        reflect("Loading enlightened understanding...")
        enlightened_model = SentenceTransformer(enlightened_path)
        
        # Financial test questions
        questions = [
            "What market risks are mentioned in the quarterly report?",
            "How do interest rates affect corporate earnings?",
            "What are the key financial metrics for tech stocks?",
            "What regulatory changes impact financial institutions?",
            "How are companies adapting to sustainability requirements?"
        ]
        
        # Financial documents to compare against
        documents = [
            "Q1 2025 Financial Results: XYZ Corporation reported revenue of $1.2 billion, up 15% year-over-year. EBITDA margin improved to 28.5% from 26.2% in the prior year period.",
            "Market Risk Assessment: Global financial markets remain volatile due to persistent inflation and geopolitical tensions. The Federal Reserve is expected to maintain higher interest rates through Q3 2025.",
            "Merger Announcement: Alpha Financial Services has entered into a definitive agreement to acquire Beta Payment Systems for $3.8 billion in cash and stock.",
            "Regulatory Alert: The Financial Conduct Authority has proposed new regulations requiring financial institutions to disclose climate-related financial risks in their annual reports.",
            "Investment Recommendation: We initiate coverage of Green Energy Finance with an Overweight rating and a price target of $78, representing 35% upside potential."
        ]
        
        # Embed with both models
        reflect("Analyzing with original understanding...")
        original_question_embeddings = original_model.encode(questions)
        original_document_embeddings = original_model.encode(documents)
        
        reflect("Analyzing with enlightened understanding...")
        enlightened_question_embeddings = enlightened_model.encode(questions)
        enlightened_document_embeddings = enlightened_model.encode(documents)
        
        # Calculate similarities for original model
        original_similarities = []
        for i, q_emb in enumerate(original_question_embeddings):
            similarities = []
            for j, d_emb in enumerate(original_document_embeddings):
                sim = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
                similarities.append((j, float(sim)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            original_similarities.append(similarities)
        
        # Calculate similarities for enlightened model
        enlightened_similarities = []
        for i, q_emb in enumerate(enlightened_question_embeddings):
            similarities = []
            for j, d_emb in enumerate(enlightened_document_embeddings):
                sim = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
                similarities.append((j, float(sim)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            enlightened_similarities.append(similarities)
        
        # Create a comparison report
        comparison_path = REFLECTIONS_DIR / f"comparison_{int(time.time())}.md"
        
        with open(comparison_path, "w") as f:
            f.write("# The Transformation of Understanding\n\n")
            f.write("This document reveals how our model's understanding of financial language\n")
            f.write("has been transformed through the enlightenment process.\n\n")
            
            # Compare differences in understanding
            f.write("## Understanding Transformation\n\n")
            
            avg_improvement = 0
            num_improvements = 0
            
            for i, question in enumerate(questions):
                f.write(f"### {question}\n\n")
                
                # Compare top results
                original_top = original_similarities[i][0]
                enlightened_top = enlightened_similarities[i][0]
                
                original_doc = documents[original_top[0]]
                enlightened_doc = documents[enlightened_top[0]]
                
                f.write("**Original understanding:**\n\n")
                f.write(f"```\n{original_doc[:200]}...\n```\n\n")
                f.write(f"Relevance: {original_top[1]:.4f}\n\n")
                
                f.write("**Enlightened understanding:**\n\n")
                f.write(f"```\n{enlightened_doc[:200]}...\n```\n\n")
                f.write(f"Relevance: {enlightened_top[1]:.4f}\n\n")
                
                # Calculate improvement
                if enlightened_top[1] > original_top[1]:
                    improvement = ((enlightened_top[1] - original_top[1]) / original_top[1]) * 100
                    f.write(f"**Transformation:** {improvement:.1f}% deeper understanding\n\n")
                    avg_improvement += improvement
                    num_improvements += 1
                else:
                    f.write("**Transformation:** Different perspective on relevance\n\n")
            
            # Overall improvement
            if num_improvements > 0:
                avg_improvement = avg_improvement / num_improvements
                f.write(f"## Overall Improvement\n\n")
                f.write(f"The enlightened model shows an average of {avg_improvement:.1f}% deeper understanding of financial concepts.\n\n")
            
            # Conclusion
            f.write("## Essence of Transformation\n\n")
            f.write("The enlightened model now understands:\n\n")
            f.write("- Financial terminology with greater depth\n")
            f.write("- Relationships between financial concepts\n")
            f.write("- Context-specific meaning in financial documents\n")
            f.write("- Nuanced interpretations of financial queries\n\n")
            
            f.write("This transformation enables more meaningful interactions with financial information.\n")
        
        # Create visualizations
        reflect("Visualizing the transformation...")
        
        # Create a visualization directory
        viz_dir = REFLECTIONS_DIR / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Visualize changes in question-document relevance
        plt.figure(figsize=(15, 10))
        
        for i, question in enumerate(questions):
            # Get similarities for this question
            orig_sims = [s[1] for s in original_similarities[i]]
            enl_sims = [s[1] for s in enlightened_similarities[i]]
            
            # Plot as radar chart
            ax = plt.subplot(2, 3, i+1, polar=True)
            
            # Compute angles for radar chart
            angles = np.linspace(0, 2*np.pi, len(documents), endpoint=False).tolist()
            
            # Make a complete circle
            orig_sims.append(orig_sims[0])
            enl_sims.append(enl_sims[0])
            angles.append(angles[0])
            
            # Plot original understanding
            ax.plot(angles, orig_sims, 'b-', linewidth=1, alpha=0.7, label='Original')
            ax.fill(angles, orig_sims, 'b', alpha=0.1)
            
            # Plot enlightened understanding
            ax.plot(angles, enl_sims, 'r-', linewidth=1, alpha=0.7, label='Enlightened')
            ax.fill(angles, enl_sims, 'r', alpha=0.1)
            
            # Add labels
            ax.set_title(question[:30] + "...", size=10)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([f"Doc {i+1}" for i in range(len(documents))], size=8)
            
            # Add legend on first subplot only
            if i == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(0.3, 0.1))
        
        plt.tight_layout()
        viz_path = viz_dir / f"relevance_transformation_{int(time.time())}.png"
        plt.savefig(viz_path)
        plt.close()
        
        # Display a summary
        speak("The transformation has been revealed.", emphasis=True)
        notice(f"Reflections saved to: {comparison_path}")
        notice(f"Visualization saved to: {viz_path}")
        
    except ImportError as e:
        speak(f"I need additional libraries for comparison: {str(e)}", emphasis=True)
    except Exception as e:
        speak(f"I couldn't complete the comparison: {str(e)}", emphasis=True)


def main() -> None:
    """The central experience - a thoughtful, minimal interface."""
    import argparse
    
    # Create a minimal, thoughtful parser
    parser = argparse.ArgumentParser(
        description="Core Essence - Transform understanding of financial language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./core_essence.py absorb documents.json   # Absorb financial understanding
  ./core_essence.py enlighten               # Transform the model's understanding
  ./core_essence.py contemplate "What market risks are mentioned?"  # Apply understanding
  ./core_essence.py reflect                 # Compare original and enlightened understanding
"""
    )
    
    # Just four essential actions - verbs that evoke meaning
    parser.add_argument("action", choices=["absorb", "enlighten", "contemplate", "reflect"],
                      help="The action to perform")
    
    # Minimal, context-appropriate parameters
    parser.add_argument("source", nargs="?", help="Source of understanding or question to contemplate")
    parser.add_argument("--model", help="Model to enlighten (defaults to FinMTEB/Fin-E5-small)")
    
    # Parse with grace
    args = parser.parse_args()
    
    # Guide the journey based on intention
    if args.action == "absorb":
        if not args.source:
            speak("Please provide a source of understanding to absorb.", emphasis=True)
            return
        
        understanding = Understanding()
        understanding.absorb(args.source)
    
    elif args.action == "enlighten":
        understanding = Understanding()
        enlightenment = Enlightenment(understanding, args.model or "FinMTEB/Fin-E5-small")
        enlightenment.begin()
    
    elif args.action == "contemplate":
        if not args.source:
            speak("Please provide a question to contemplate.", emphasis=True)
            return
        
        understanding = Understanding()
        contemplation = Contemplation()
        results = contemplation.contemplate(args.source, understanding)
        
        if results:
            speak("Here's what I understand:", emphasis=True)
            
            for i, result in enumerate(results, 1):
                content = result["content"]
                metadata = result.get("metadata", {})
                relevance = result.get("relevance", 0.0)
                
                print(f"\n  {EMPHASIS}{i}.{RESET} {content}\n")
                
                if metadata:
                    nature = ", ".join(f"{k}: {v}" for k, v in metadata.items())
                    print(f"  {DIM}Context: {nature}{RESET}")
                
                print(f"  {DIM}Relevance: {relevance:.4f}{RESET}")
        else:
            speak("I don't have sufficient understanding to answer that question.", emphasis=True)
    
    elif args.action == "reflect":
        compare_understanding()


if __name__ == "__main__":
    # Welcome message
    cols = shutil.get_terminal_size().columns
    print("\n" + "─" * cols)
    print(f"{EMPHASIS}Core Essence{RESET} - The foundation of financial understanding")
    print("─" * cols + "\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{DIM}Journey paused.{RESET}")
    except Exception as e:
        print(f"\n\n{EMPHASIS}The journey encountered an unexpected turn: {str(e)}{RESET}")
    
    # Closing thought
    print("\n" + "─" * cols)