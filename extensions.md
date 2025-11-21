Areas of further development:
- Alignments should be possible on a per-node basis (containers and arrays). Useful to eg, align certain nodes with pages but not others to reduce cross-page loads. Separate pages for different CPU/GPU cores may be relevant as well.
- A 'compiler' that can take python code or a reduced representation (to include other languages) map all mallocs and arrays, and autogen a buffer map also producing a template for the top level fixed and variable dimension dependencies.
- An extension that procedurally rewrites the code to utilize the static arrays/buffer.
- Or simply an LLM pipeline, agentic flow that can do both.
