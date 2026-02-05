//! Anthropic API client and Clankers integration
//!
//! # Example
//! ```
//! use clankers::providers::anthropic;
//!
//! let client = anthropic::Anthropic::new("YOUR_API_KEY");
//!
//! let sonnet = client.completion_model(anthropic::CLAUDE_3_5_SONNET);
//! ```

pub mod client;
pub mod completion;
pub mod decoders;
pub mod streaming;

pub use client::{Client, ClientBuilder};
