# Paper Downloader 📜

This repository provides a simple tool for downloading free scientific papers based on a structured
index of papers stored in `index.md`. The repository contains the following files:

## ✨ Features

- 💾 Download papers from URLs specified in `index.md`.
- 🗂️ Organize downloaded papers into folders based on categories.
- ⚡ Easy-to-use and lightweight tool for researchers and students.

## 🚀 Getting Started 

### Prerequisites

Ensure you have a reasonable version of Python installed on your system.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Kripner/papiry
   cd papiry
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Update the `index.md` file to include the papers you want to download. The format of `index.md` is as follows:

   ```markdown
   # [CategoryName] Optional description

   - [PaperName] Paper Title: <URL>
   ```

   Example:

   ```markdown
   # [games] Two-player zero-sum games

   - [AlphaGoZero] Mastering the game of Go without human knowledge: https://ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/nature-go.pdf
   - [AlphaZero] Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm: https://arxiv.org/abs/1712.01815
   ```

2. Run the `download.py` script:

   ```bash
   python download.py
   ```

3. The script will create a folder for each category and download the corresponding papers into the respective folders.

## Example

Given the following `index.md`:

```markdown
# [games] Two-player zero-sum games

- [AlphaGoZero] Mastering the game of Go without human knowledge: https://ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/nature-go.pdf
- [AlphaZero] Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm: https://arxiv.org/abs/1712.01815
```

Running the script will create the following folder structure:

```
output/
├── games/
   ├── AlphaGoZero.pdf
   └── AlphaZero.pdf
```

## License

This project is released to the public domain. You are free to use, modify, and distribute this code without any restrictions.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential improvements.


