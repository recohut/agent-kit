# agent-kit

Runnable agent examples from the Recohut blog, with shared Python utilities.

Each article has a directory under `examples/<article-slug>/`. Keep its entry
point, setup instructions and synthetic fixtures there. Put reusable code in
`utils/` and import it by module name.

From the repository root:

```sh
python -m examples.hello_agent.main
```

For a separate checkout, install the utilities from a reviewed commit with
`pip install 'agent-kit @ git+https://github.com/recohut/agent-kit.git@COMMIT_SHA'`.
Then import them with `from utils import ...`. Pin a full commit SHA.

Use synthetic examples and placeholder configuration. Review every diff before
pushing to a public remote: a pull request check runs after the push and cannot
prevent exposure on a public branch. Scan locally before pushing.
