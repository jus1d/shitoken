# shitoken

## Quick start

Generate a BPE table

```bash
$ shitoken generate --corpus ./shakespeare.txt --output ./shakespeare.bpe --merges 500
```

Tokenize something

```bash
$ shitoken tokenize --table data/shakespeare.bpe --raw "Tokenize me"
84:"T" 111:"o" 107:"k" 101:"e" 110:"n" 105:"i" 122:"z" 101:"e" 32:" " 109:"m" 101:"e"
```
