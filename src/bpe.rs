use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy, PartialOrd, Ord)]
pub struct Pair {
    pub l: u32,
    pub r: u32,
}

pub struct Bpe {
    pub merges: Vec<(Pair, u32)>,
    pub vocab: HashMap<u32, String>,
}

impl Bpe {
    /// Train BPE on `text` for up to `num_merges` iterations.
    ///
    /// Maintains an implicit doubly-linked list over the token sequence so
    /// each merge visits only the O(k) positions of the chosen pair instead
    /// of scanning the whole corpus. Pair frequencies are updated
    /// incrementally; the best pair is found via a max-heap with lazy
    /// deletion of stale entries.
    pub fn train(text: &str, num_merges: usize) -> Self {
        let chars: Vec<u32> = text.chars().map(|c| c as u32).collect();
        let n = chars.len();

        let mut vocab: HashMap<u32, String> = text
            .chars()
            .map(|c| (c as u32, c.to_string()))
            .collect();

        let mut merges = Vec::new();
        let mut next_id = vocab.keys().copied().max().unwrap_or(0) + 1;

        if n < 2 {
            return Bpe { merges, vocab };
        }

        let mut vals: Vec<u32> = chars;
        let mut nxt: Vec<usize> = (1..=n).collect();
        let mut prv: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut alive: Vec<bool> = vec![true; n];

        let mut pair_freq: HashMap<Pair, usize> = HashMap::new();
        let mut pair_pos: HashMap<Pair, HashSet<usize>> = HashMap::new();

        let mut i = 0;
        loop {
            let j = nxt[i];
            if j >= n {
                break;
            }
            let p = Pair { l: vals[i], r: vals[j] };
            *pair_freq.entry(p).or_insert(0) += 1;
            pair_pos.entry(p).or_default().insert(i);
            i = j;
        }

        let mut heap: BinaryHeap<(usize, Pair)> =
            pair_freq.iter().map(|(&p, &c)| (c, p)).collect();

        for _ in 0..num_merges {
            let best = loop {
                let Some((count, pair)) = heap.pop() else {
                    return Bpe { merges, vocab };
                };
                let current = pair_freq.get(&pair).copied().unwrap_or(0);
                if current == 0 {
                    continue;
                }
                if count == current {
                    break pair;
                }
            };

            let new_id = next_id;
            next_id += 1;

            let l_str = vocab.get(&best.l).cloned().unwrap_or_default();
            let r_str = vocab.get(&best.r).cloned().unwrap_or_default();
            vocab.insert(new_id, format!("{l_str}{r_str}"));

            let positions = pair_pos.remove(&best).unwrap_or_default();
            pair_freq.remove(&best);

            for pos in &positions {
                let pos = *pos;

                if !alive[pos] {
                    continue;
                }
                let right = nxt[pos];
                if right >= n || !alive[right] {
                    continue;
                }
                if vals[pos] != best.l || vals[right] != best.r {
                    continue;
                }

                let left = prv[pos];
                if left < n && alive[left] {
                    update_pair(
                        &mut pair_freq,
                        &mut pair_pos,
                        &mut heap,
                        Pair { l: vals[left], r: best.l },
                        left,
                        Pair { l: vals[left], r: new_id },
                        left,
                    );
                }

                let rr = nxt[right];
                if rr < n && alive[rr] {
                    update_pair(
                        &mut pair_freq,
                        &mut pair_pos,
                        &mut heap,
                        Pair { l: best.r, r: vals[rr] },
                        right,
                        Pair { l: new_id, r: vals[rr] },
                        pos,
                    );
                }

                vals[pos] = new_id;
                alive[right] = false;
                nxt[pos] = nxt[right];
                if nxt[right] < n {
                    prv[nxt[right]] = pos;
                }
            }

            merges.push((best, new_id));
        }

        Bpe { merges, vocab }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<u32> = text.chars().map(|c| c as u32).collect();
        for &(pair, new_id) in &self.merges {
            tokens = merge_pass(&tokens, pair, new_id);
        }
        tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|&id| self.token_str(id)).collect()
    }

    pub fn token_str(&self, id: u32) -> String {
        self.vocab.get(&id).cloned().unwrap_or_else(|| {
            char::from_u32(id)
                .map(|c| c.to_string())
                .unwrap_or_default()
        })
    }

    // Binary format:
    //   magic:      8 bytes  "BPE\x00\x00\x00\x00\x01"
    //   num_vocab:  u64 le
    //   vocab[i]:   u32 le id, u32 le str_len, str_len bytes (UTF-8)
    //   num_merges: u64 le
    //   merges[i]:  u32 le pair.l, u32 le pair.r, u32 le new_id

    pub fn save(&self, path: &str) -> io::Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);

        w.write_all(b"BPE\x00\x00\x00\x00\x01")?;

        w.write_all(&(self.vocab.len() as u64).to_le_bytes())?;
        for (&id, s) in &self.vocab {
            let b = s.as_bytes();
            w.write_all(&id.to_le_bytes())?;
            w.write_all(&(b.len() as u32).to_le_bytes())?;
            w.write_all(b)?;
        }

        w.write_all(&(self.merges.len() as u64).to_le_bytes())?;
        for &(pair, new_id) in &self.merges {
            w.write_all(&pair.l.to_le_bytes())?;
            w.write_all(&pair.r.to_le_bytes())?;
            w.write_all(&new_id.to_le_bytes())?;
        }

        w.flush()
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let mut data = Vec::new();
        File::open(path)?.read_to_end(&mut data)?;

        let mut cur = 0usize;

        macro_rules! read_bytes {
            ($n:expr) => {{
                let end = cur + $n;
                if end > data.len() {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated BPE file"));
                }
                let slice = &data[cur..end];
                cur = end;
                slice
            }};
        }
        macro_rules! read_u32 {
            () => {
                u32::from_le_bytes(read_bytes!(4).try_into().unwrap())
            };
        }
        macro_rules! read_u64 {
            () => {
                u64::from_le_bytes(read_bytes!(8).try_into().unwrap())
            };
        }

        let magic = read_bytes!(8);
        if magic != b"BPE\x00\x00\x00\x00\x01" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic bytes"));
        }

        let num_vocab = read_u64!() as usize;
        let mut vocab = HashMap::with_capacity(num_vocab);
        for _ in 0..num_vocab {
            let id = read_u32!();
            let len = read_u32!() as usize;
            let s = std::str::from_utf8(read_bytes!(len))
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                .to_string();
            vocab.insert(id, s);
        }

        let num_merges = read_u64!() as usize;
        let mut merges = Vec::with_capacity(num_merges);
        for _ in 0..num_merges {
            let l = read_u32!();
            let r = read_u32!();
            let new_id = read_u32!();
            merges.push((Pair { l, r }, new_id));
        }

        Ok(Bpe { merges, vocab })
    }
}

/// Remove `old` pair at `old_pos`, add `new` pair at `new_pos`, push to heap.
fn update_pair(
    freq: &mut HashMap<Pair, usize>,
    pos: &mut HashMap<Pair, HashSet<usize>>,
    heap: &mut BinaryHeap<(usize, Pair)>,
    old: Pair,
    old_pos: usize,
    new: Pair,
    new_pos: usize,
) {
    if let Some(c) = freq.get_mut(&old) {
        *c = c.saturating_sub(1);
    }
    if let Some(set) = pos.get_mut(&old) {
        set.remove(&old_pos);
    }

    let c = freq.entry(new).or_insert(0);
    *c += 1;
    pos.entry(new).or_default().insert(new_pos);
    heap.push((*c, new));
}

fn merge_pass(tokens: &[u32], pair: Pair, new_id: u32) -> Vec<u32> {
    let mut result = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == pair.l && tokens[i + 1] == pair.r {
            result.push(new_id);
            i += 2;
        } else {
            result.push(tokens[i]);
            i += 1;
        }
    }
    result
}
