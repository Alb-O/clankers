use clankers::agent::stream_to_stdout;
use clankers::prelude::*;
use clankers::providers::xai;
use clankers::streaming::StreamingPrompt;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
	// Create streaming agent with a single context prompt
	let agent = xai::Client::from_env()
		.agent(xai::completion::GROK_3_MINI)
		.preamble("Be precise and concise.")
		.temperature(0.5)
		.build();

	// Stream the response and print chunks as they arrive
	let mut stream = agent
		.stream_prompt("When and where and what type is the next solar eclipse?")
		.await;

	let _ = stream_to_stdout(&mut stream).await?;
	println!();

	Ok(())
}
