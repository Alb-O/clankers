use clankers::agent::stream_to_stdout;
use clankers::message::Message;
use clankers::prelude::*;
use clankers::providers::{self, openai};
use clankers::streaming::StreamingChat;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
	// Create OpenAI client
	let client = providers::openai::Client::from_env();

	// Create agent with a single context prompt
	let comedian_agent = client
		.agent(openai::completion::types::GPT_4)
		.preamble("You are a comedian here to entertain the user using humour and jokes.")
		.build();

	let messages = vec![
		Message::user("Tell me a joke!"),
		Message::assistant("Why did the chicken cross the road?\n\nTo get to the other side!"),
	];

	// Prompt the agent and print the response
	let mut stream = comedian_agent.stream_chat("Entertain me!", messages).await;

	let res = stream_to_stdout(&mut stream).await.unwrap();

	println!("Response: {res:?}");

	Ok(())
}
