# DocsOp

On any new push, a workflow is triggered to automatically generate [`API documentation`](../api-documentation/index.md) directly from the source code via `pydoc-markdown`. 

A `VitePress` build is then generated with all the contents from `docs` directly and published to Github Pages.

You can view the workflow [`here`](../../.github/workflows/deploy-docs.yml)

