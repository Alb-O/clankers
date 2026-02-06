//! Ollama API client and Clankers integration
//!
//! # Example
//! ```rust,ignore
//! use clankers::client::{Nothing, CompletionClient};
//! use clankers::completion::Prompt;
//! use clankers::providers::ollama;
//!
//! // Create a new Ollama client (defaults to http://localhost:11434)
//! // In the case of ollama, no API key is necessary, so we use the `Nothing` struct
//! let client: ollama::Client = ollama::Client::new(Nothing).unwrap();
//!
//! // Create an agent with a preamble
//! let comedian_agent = client
//!     .agent("qwen2.5:14b")
//!     .preamble("You are a comedian here to entertain the user using humour and jokes.")
//!     .build();
//!
//! // Prompt the agent and print the response
//! let response = comedian_agent.prompt("Entertain me!").await?;
//! println!("{response}");
//!
//! // Create an embedding model using the "all-minilm" model
//! let emb_model = client.embedding_model("all-minilm", 384);
//! let embeddings = emb_model.embed_texts(vec![
//!     "Why is the sky blue?".to_owned(),
//!     "Why is the grass green?".to_owned()
//! ]).await?;
//! println!("Embedding response: {:?}", embeddings);
//!
//! // Create an extractor if needed
//! let extractor = client.extractor::<serde_json::Value>("llama3.2").build();
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod message;

pub use client::{Client, ClientBuilder};
pub use completion::{CompletionModel, CompletionResponse, StreamingCompletionResponse};
pub use embedding::{EmbeddingModel, EmbeddingResponse};
pub use message::*;

// ================================================================
// Ollama Embedding Model Constants
// ================================================================

pub const ALL_MINILM: &str = "all-minilm";
pub const NOMIC_EMBED_TEXT: &str = "nomic-embed-text";

// ================================================================
// Ollama Completion Model Constants
// ================================================================

pub const LLAMA3_2: &str = "llama3.2";
pub const LLAVA: &str = "llava";
pub const MISTRAL: &str = "mistral";
