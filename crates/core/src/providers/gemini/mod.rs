//! Google Gemini API client and Clankers integration
//!
//! # Example
//! ```
//! use clankers::providers::gemini;
//!
//! let client = gemini::Client::new("YOUR_API_KEY");
//!
//! let gemini_embedding_model = client.embedding_model(gemini::EMBEDDING_001);
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod streaming;
pub mod transcription;

pub use client::Client;
pub use completion::CompletionModel;
pub use embedding::{EMBEDDING_001, EMBEDDING_004, EmbeddingModel};

pub mod api_types;
