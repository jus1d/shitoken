mod bpe;
use bpe::Bpe;
use clap::{Parser, Subcommand};
use std::fs;
use std::io::{self, BufRead, Read, Write};

const TOKEN_COLORS: &[(&str, &str)] = &[
    ("\x1b[48;2;255;213;183m", "\x1b[38;2;0;0;0m"), // peach
    ("\x1b[48;2;183;247;195m", "\x1b[38;2;0;0;0m"), // mint
    ("\x1b[48;2;183;220;255m", "\x1b[38;2;0;0;0m"), // sky
    ("\x1b[48;2;255;183;227m", "\x1b[38;2;0;0;0m"), // pink
    ("\x1b[48;2;220;195;255m", "\x1b[38;2;0;0;0m"), // lavender
    ("\x1b[48;2;255;245;183m", "\x1b[38;2;0;0;0m"), // yellow
];

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";

#[derive(Parser)]
#[command(name = "shitoken", about = "shitoken is a shitty BPE tokenizer", arg_required_else_help = true)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train BPE on a corpus and save the table to a file.
    Generate {
        /// Corpus file to train on.
        #[arg(short, long)]
        corpus: String,

        /// Output file for the BPE table.
        #[arg(short, long)]
        output: String,

        /// Number of BPE merge operations.
        #[arg(short, long, default_value_t = 100)]
        merges: usize,
    },

    /// Tokenize text using a saved BPE table (or train on the fly).
    Tokenize {
        /// Text to tokenize. Reads from stdin if omitted.
        text: Option<String>,

        /// Path to a saved BPE table file.
        #[arg(short, long)]
        table: Option<String>,

        /// Train on this corpus file instead of loading a table.
        #[arg(long)]
        train: Option<String>,

        /// Number of BPE merge operations (only used with --train).
        #[arg(short, long, default_value_t = 100)]
        merges: usize,

        /// Output token IDs separated by spaces instead of colored text.
        #[arg(short, long)]
        raw: bool,

        /// Interactive REPL mode - tokenize lines on demand.
        #[arg(short, long)]
        interactive: bool,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Command::Generate { corpus, output, merges } => {
            let text = read_file(&corpus);
            eprintln!("Training on {} chars, {} merges...", text.chars().count(), merges);
            let bpe = Bpe::train(&text, merges);
            if let Err(e) = bpe.save(&output) {
                eprintln!("ERROR: could not save `{output}`: {e}");
                std::process::exit(1);
            }
            eprintln!("Saved {} vocab entries, {} merges -> {output}", bpe.vocab.len(), bpe.merges.len());
        }

        Command::Tokenize { text, table, train, merges, raw, interactive } => {
            let bpe = if table.is_none() && train.is_none() {
                match &text {
                    Some(t) => Bpe::train(t, merges),
                    None => {
                        eprintln!("ERROR: provide --table, --train, or text argument");
                        std::process::exit(1);
                    }
                }
            } else {
                load_bpe(table, train, merges)
            };

            if interactive {
                eprintln!("{DIM}vocab {} tokens, {} merges -- Ctrl+D to quit{RESET}", bpe.vocab.len(), bpe.merges.len());
                let stdin = io::stdin();
                loop {
                    print!("{BOLD}>{RESET} ");
                    io::stdout().flush().ok();

                    let mut line = String::new();
                    match stdin.lock().read_line(&mut line) {
                        Ok(0) => break,
                        Ok(_) => {}
                        Err(e) => { eprintln!("ERROR: {e}"); break; }
                    }

                    let text = line.trim_end_matches('\n').trim_end_matches('\r');
                    if text.is_empty() { continue; }

                    let tokens = bpe.encode(text);
                    if raw {
                        print_raw(&bpe, &tokens);
                    } else {
                        print_tokens(&bpe, &tokens, text);
                        print_stats(&tokens, text);
                    }
                }
            } else {
                let text = match text {
                    Some(t) => t,
                    None => {
                        let mut buf = String::new();
                        io::stdin().read_to_string(&mut buf).expect("failed to read stdin");
                        buf
                    }
                };

                let tokens = bpe.encode(&text);
                if raw {
                    print_raw(&bpe, &tokens);
                } else {
                    print_tokens(&bpe, &tokens, &text);
                    print_stats(&tokens, &text);
                }
            }
        }
    }
}

fn load_bpe(table: Option<String>, train: Option<String>, merges: usize) -> Bpe {
    match (table, train) {
        (Some(_), Some(_)) => {
            eprintln!("ERROR: --table and --train are mutually exclusive");
            std::process::exit(1);
        }
        (Some(path), None) => match Bpe::load(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("ERROR: could not load `{path}`: {e}");
                std::process::exit(1);
            }
        },
        (None, Some(path)) => {
            let corpus = read_file(&path);
            Bpe::train(&corpus, merges)
        }
        (None, None) => {
            eprintln!("ERROR: provide --table or --train");
            std::process::exit(1);
        }
    }
}

fn read_file(path: &str) -> String {
    match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: could not read `{path}`: {e}");
            std::process::exit(1);
        }
    }
}

fn print_raw(bpe: &Bpe, tokens: &[u32]) {
    // Format: <id>:"<string>" per token, space-separated.
    let parts: Vec<String> = tokens
        .iter()
        .map(|&id| {
            let s = bpe.token_str(id);
            let escaped = s
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            format!("{id}:\"{}\"", escaped)
        })
        .collect();
    println!("{}", parts.join(" "));
}

fn print_tokens(bpe: &Bpe, tokens: &[u32], original: &str) {
    println!();
    for (i, &id) in tokens.iter().enumerate() {
        let s = bpe.token_str(id);
        let (bg, fg) = TOKEN_COLORS[i % TOKEN_COLORS.len()];
        let displayed = s.replace('\n', "↵\n");
        print!("{bg}{fg}{displayed}{RESET}");
    }

    if bpe.decode(tokens) != original {
        eprintln!("\n{BOLD}WARNING: round-trip decode mismatch{RESET}");
    }
}

fn print_stats(tokens: &[u32], original: &str) {
    let char_count = original.chars().count();
    let token_count = tokens.len();
    let ratio = token_count as f64 / char_count as f64 * 100.0;

    println!("\n");
    println!(
        "{DIM}chars{RESET}   {BOLD}{char_count}{RESET}   \
         {DIM}tokens{RESET}   {BOLD}{token_count}{RESET}   \
         {DIM}ratio{RESET}   {BOLD}{ratio:.1}%{RESET}"
    );
}
