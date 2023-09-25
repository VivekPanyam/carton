# Contributing

We welcome contributions! These can take many forms such as docs, answering questions, and helping others. You can also add support for new ML frameworks or programming languages.

Check out [`ARCHITECTURE.md`](./ARCHITECTURE.md) for an overview of the structure of the project and how Carton works.

## What if I don't know Rust?

Rust is a powerful language, but it does come with a learning curve. [This](https://fasterthanli.me/articles/a-half-hour-to-learn-rust) is a very good guide on how to read Rust code.

That said, there are ways to contribute without learning Rust:

* The documentation website in `docs/website` is a good place to help out
* Writing bindings for other languages on top of existing ones (for languages without good interop with Rust directly). For example, creating:
    * Golang, Java, Scala, or Kotlin  bindings on top of the C or C++ bindings
        * Note: The C and C++ bindings are currently not implemented, but they can be prioritized if someone is interested in building other bindings on top of them.
* Improving the foreign language side of existing bindings (e.g. python tests, etc.)

## New Issues

If you're interested in implementing a new feature or bugfix:

1. Please create [an issue](https://github.com/VivekPanyam/carton/issues) describing the proposed feature/bugfix.

2. We'll discuss the implementation plan and design. If everything looks good, go ahead and implement it.

3. Once you're ready to submit your change for review, create a pull request (see GitHub's instructions on [forking a repo](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) and [creating a pull request from a fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)).

## Open issues

If you're interested in implementing a feature or fixing a bug described by an [existing issue](https://github.com/VivekPanyam/carton/issues):

1. Comment on the issue saying you'd like to work on it and provide some detail about your proposed solution.

2. We'll discuss the implementation plan and design. If everything looks good, go ahead and implement it.

3. Once you're ready to submit your change for review, create a pull request (see GitHub's instructions on [forking a repo](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) and [creating a pull request from a fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)).


Some issues are tagged "good first issue". These are generally a good place to start if you're looking for a way to contribute.

## Adding support for a new ML framework

Copying the `source/carton-runner-noop` crate is a good place to start. It implements a basic runner that just returns the inputs. You can also look through the code for other runners.

See tests in `source/carton-runner-*/tests` for examples of installing runners and testing how they integrate with Carton.

Make sure to create an issue as described above to avoid duplicate work in case others are also working on adding support for the same framework.

## Adding support for a new programming language

A language binding crate is basically a bridge between `[some programming language]` and the `carton` Rust crate.

For languages with good Rust interop or some sort of bindgen crates, it's recommended to use those to help build bindings. The goal of the bindings is to prioritize UX and make it feel like the library "belongs" in that language.

For other languages, it may make more sense to build on top of C and C++ bindings (see the "What if I don't know Rust?" section above).

See the existing bindings under `source/carton-bindings-*`.

Make sure to create an issue as described above to avoid duplicate work in case others are also working on adding support for the same language.

## Guidelines

To streamline the PR review process, please:

- Write clear and concise titles and descriptions
- Include tests in your PR
- Include comments in your code
- Keep PRs small. These tend to be easier to review.