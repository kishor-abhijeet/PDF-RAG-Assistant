import arxiv

client = arxiv.Client()

search = arxiv.Search(
    query="functions of kernel",
    max_results=2,
)

results = client.results(search)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Title:", result.title)
    print("Authors:", ", ".join(author.name for author in result.authors))
    print("Summary:", result.summary)