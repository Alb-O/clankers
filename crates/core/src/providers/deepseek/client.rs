use super::completion::CompletionModel;
use crate::client::{self, BearerAuth, Capable, Nothing, ProviderClient};
use crate::providers::openai_compat::{self, OpenAiCompat, PBuilder};

const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com";

#[derive(Debug, Default, Clone, Copy)]
pub struct DeepSeek;

impl OpenAiCompat for DeepSeek {
	const PROVIDER_NAME: &'static str = "deepseek";
	const BASE_URL: &'static str = DEEPSEEK_API_BASE_URL;
	const API_KEY_ENV: &'static str = "DEEPSEEK_API_KEY";
	const VERIFY_PATH: &'static str = "/user/balance";
	const COMPLETION_PATH: &'static str = "/chat/completions";

	type BuilderState = ();
	type Completion<H> = Capable<CompletionModel<H>>;
	type Embeddings<H> = Nothing;
	type Transcription<H> = Nothing;
	#[cfg(feature = "image")]
	type ImageGeneration<H> = Nothing;
	#[cfg(feature = "audio")]
	type AudioGeneration<H> = Nothing;
}

pub type Client<H = reqwest::Client> = client::Client<DeepSeek, H>;
pub type ClientBuilder<H = reqwest::Client> =
	client::ClientBuilder<PBuilder<DeepSeek>, BearerAuth, H>;

impl ProviderClient for Client {
	type Input = String;

	fn from_env() -> Self {
		openai_compat::default_from_env::<DeepSeek>()
	}

	fn from_val(input: Self::Input) -> Self {
		Self::new(&input).unwrap()
	}
}
