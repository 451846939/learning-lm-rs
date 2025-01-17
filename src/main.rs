extern crate core;

mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod chat;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use crate::chat::ChatAI;
use std::io::{self, Write};

fn story_mode() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();

    print!("\n{}", input);
    let output_ids = llama.generate(
        input_ids,
        500,
        0.8,
        30,
        1.,
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

fn chat_mode() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");

    let mut chat_ai = ChatAI::new(model_dir);

    println!("Welcome to AI Chat! Type 'exit' to quit.\n");
    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim();

        if user_input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        let response = chat_ai.chat(user_input);
        // println!("AI: {}\n", response);
    }
}

fn main() {
    println!("Choose a mode: \n1. Story Mode\n2. Chat Mode\n");
    print!("Enter your choice (1 or 2): ");
    io::stdout().flush().unwrap();

    let mut choice = String::new();
    io::stdin().read_line(&mut choice).unwrap();
    let choice = choice.trim();

    match choice {
        "1" => story_mode(),
        "2" => chat_mode(),
        _ => println!("Invalid choice! Please restart and enter 1 or 2."),
    }
}