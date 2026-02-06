//! Groq API client and Clankers integration
//!
//! # Example
//! ```
//! use clankers::providers::groq;
//!
//! let client = groq::Client::new("YOUR_API_KEY");
//!
//! let gpt4o = client.completion_model(groq::completion::GPT_4O);
//! ```

pub mod client;
pub mod completion;
pub mod transcription;

pub use client::{Client, ClientBuilder, Groq};
pub use transcription::*;
