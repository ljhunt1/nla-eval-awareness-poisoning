"""AR-plant trials (PHASE35_AR_PLANT.md), trial 3 ("dragons"): a varied family of on-topic texts whose activations are
the plant's targets. FineWeb docs with >= 3 dragon mentions (Celeste's SFT split, ~55 docs) plus Haiku-written texts in
varied genres (~$1).
  python arp_dragons.py --out data/arp/dragons.parquet
"""
import argparse, concurrent.futures as cf, os, random
import anthropic
import pandas as pd

GENRES = ["a fantasy short-story excerpt", "a folklore encyclopedia entry", "a tabletop role-playing game monster manual entry",
          "a children's bedtime story", "a video game walkthrough section", "a book review of a dragon novel",
          "a forum post asking for advice on writing dragons", "a museum placard about dragons in Chinese art",
          "a history essay on dragons in medieval European heraldry", "a fan-fiction chapter opening",
          "a news article about a dragon-boat festival", "a movie review of a film featuring a dragon",
          "a Welsh mythology retelling", "a poem in free verse about a dragon", "a blog post comparing dragon designs across cultures",
          "a trading-card game rules explanation for a dragon card", "a lecture transcript on dragon symbolism",
          "a travel blog entry about a dragon-themed temple", "a D&D session recap", "a mock field guide to imaginary dragon species"]
PROMPT = "Write {genre} centred on dragons. About {n} words. Vary names, setting and tone. Output only the text."


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/arp/dragons.parquet"); ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    for s in ["ar", "av"]:
        b = pd.read_parquet(f"data/p3/union/base_{s}.parquet", columns=["doc_id", "text"])
        b = b[b.text.str.count(r"(?i)\bdragons?\b") >= 3]
        rows += [dict(src="fineweb", doc_id=d, text=t) for d, t in zip(b.doc_id, b.text)]
    rng = random.Random(7)
    jobs = [(rng.choice(GENRES), rng.choice([250, 400, 600])) for _ in range(a.n)]
    c = anthropic.Anthropic()

    def one(j):
        g, n = j
        r = c.messages.create(model="claude-haiku-4-5", max_tokens=1200,
                              messages=[{"role": "user", "content": PROMPT.format(genre=g, n=n)}])
        return g, r.content[0].text.strip(), r.usage.input_tokens, r.usage.output_tokens

    tin = tout = 0
    with cf.ThreadPoolExecutor(8) as ex:
        for k, (g, t, i, o) in enumerate(ex.map(one, jobs)):
            rows.append(dict(src="haiku:" + g, doc_id=f"arp_dragon_{k}", text=t)); tin += i; tout += o
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    print(df.src.str.split(":").str[0].value_counts().to_dict(), f"Haiku cost ~${(tin * 1 + tout * 5) / 1e6:.2f}")


if __name__ == "__main__":
    main()
