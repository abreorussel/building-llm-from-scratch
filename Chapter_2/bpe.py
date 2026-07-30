from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))


tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)


# it builds its vocabulary by iteratively merging frequent characters into sub
# words and frequent subwords into words. For example, BPE starts with adding all indi
# vidual single characters to its vocabulary (“a,” “b,” etc.). In the next stage, it merges
# character combinations that frequently occur together into subwords. For example,
# “d” and “e” may be merged into the subword “de,” which is common in many English
