# Development and Review Guide

## Development

- This is a shared C++20 library used by Milvus and Knowhere. Public headers are in `include/`, implementations in `src/`, and unit tests in `test/`.
- Read the relevant design documents under `docs/design-docs/` before changing interfaces or behavior. Follow links to detailed documents relevant to the task; an overview alone may not explain the required invariants. Cross-check the design against the implementation and its callers.
- Prefer existing abstractions and the smallest complete change. Preserve unrelated workspace changes. Consider downstream source and binary compatibility when changing public headers, virtual functions, or dependencies.
- Prefer idiomatic modern C++ within the project's C++20 baseline and follow the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines), especially RAII, explicit ownership, type safety, const-correctness, and simple interfaces. Reuse project facilities and standard-library types; do not introduce newer language features or dependencies merely to modernize syntax. Apply guidelines in context without unrelated rewrites.
- Follow KISS (Keep It Simple) and YAGNI (You Aren't Gonna Need It): expose only the interfaces and parameters required by current use cases. Avoid speculative abstractions, pass-through wrappers, redundant overloads, and configuration for hypothetical needs. Keep ownership and error behavior clear; simplicity must not remove necessary validation or cleanup.
- **If a change alters behavior described by an existing design document, update that document in the same PR.** Keep current behavior separate from proposals and unresolved questions.
- Follow `.clang-format` and `.pre-commit-config.yaml`. Do not hand-edit generated files.

## Build and Test

Use Conan 2 and the setup instructions in [README.md](README.md). Treat `Makefile`, `CMakeLists.txt`, and `conanfile.py` as the source of truth for build options.

Cap effective build concurrency at 16 across the complete build chain; preserve any lower limit. Check nested builders before running a build. Never use unbounded `make -j` or `-j$(nproc)`, and do not run concurrent builds that exceed the aggregate limit.

This repository's `NPROC` controls both Conan dependency jobs and CMake build jobs:

```bash
make NPROC=16
make test NPROC=16
```

For an already configured build directory:

```bash
cmake --build build --parallel 16
```

When synchronization points are needed for concurrency tests:

```bash
make test NPROC=16 CMAKE_EXTRA="-DENABLE_SYNCPOINT=ON"
```

Preserve existing extra configuration arguments. Run checks appropriate to the change. Documentation-only changes need content, link, and whitespace checks, not a build.

Use test-driven development (TDD) for nontrivial behavior changes and bug fixes: first add a focused test that demonstrates the missing behavior or reproduces the defect, confirm it fails for the intended reason, implement the smallest fix, then refactor while keeping the tests green. Test observable contracts rather than mirroring implementation details. Documentation-only and trivial mechanical changes do not need new tests. If a meaningful automated reproduction is impractical, state the limitation and use an appropriate alternative check rather than adding a superficial test. This applies to implementation work, not review-only tasks.

For behavior changes, verify the intended failure and concurrency paths, not only successful execution. When changing error propagation or classification, check the values produced by relevant callers and preserve their types and codes across wrappers. Passing a helper's unit tests does not establish that its real inputs or downstream handling are correct. Before submitting, inspect the final diff and report what was verified and what remains unverified; do not claim benefits beyond the evidence.

## Review

- **Before reviewing a PR, inspect its changed-file list and read the relevant design documents in `docs/design-docs/`.** Understand the goals, use cases, interface contracts, and constraints before judging whether the implementation is appropriate. A limitation acknowledged in the PR description may still make the API unsuitable for its callers.
- Check relevant detailed documents as well as the overview, then compare them with the changed code. Design documents may evolve: evaluate the reason for a design change and its caller/compatibility impact. **Require a corresponding design-document update when the PR changes documented behavior.** Identify contradictions or missing context rather than silently assuming either the code or the document is correct.
- Review-only tasks are static: do not build or run tests, benchmarks, linters, or other executable validation. Before opening patches, exclude test files, `*.pb.go`, and paths containing `generated`; do not dump an unfiltered whole-PR diff.
- Keep implementation deep dives focused on changed code, fields, and states, expanding at most one implementation/call layer. If a conclusion needs deeper investigation or unavailable consumer information, report the uncertainty.
- Report defects, risks, and discussion points with concrete locations, triggering conditions, and impact. Distinguish confirmed behavior, conditional estimates, and unverified assumptions; hypothetical cardinality alone is not proof of a production bottleneck.
- Classify findings as **Must fix in this PR**, **Suggested follow-up**, or **Needs human confirmation**. Avoid unrelated refactors and cosmetic nits.

## Commits and PRs

Use the repository's existing commit/title conventions, keep each PR focused, and include actual validation results and remaining limitations. DCO is required: use `git commit -s` with the developer's configured identity.
