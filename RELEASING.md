# Releasing Vox MCP

The public package is `vox-mcp` on PyPI, registered as
`io.github.linxule/vox-mcp` in the MCP Registry. The private `vox` development
repository is not a public distribution source.

1. Update `pyproject.toml`, both versions in `server.json`, and `CHANGELOG.md`.
   Run `uv lock`, the CI checks, `uv build`, and `mcp-publisher validate`.
2. Merge the tested release commit, then create and push an immutable `vX.Y.Z` tag.
3. Run `.github/workflows/publish.yml` with `release_tag=vX.Y.Z`. Its default
   `publish=false` validates and builds without uploading. Set `publish=true`
   to publish the validated artifacts to PyPI.
4. Verify the exact version at `https://pypi.org/pypi/vox-mcp/X.Y.Z/json`, then
   run `mcp-publisher publish server.json` from the tagged checkout using an
   existing MCP Registry login. Wait for PyPI propagation before registration.
5. Verify the exact MCP Registry version, test initialization and tool discovery
   from the published wheel in an isolated environment, and publish the GitHub
   release with the changelog and distribution artifacts.

PyPI trusted publishing must be configured on the existing `vox-mcp` project:
GitHub owner `linxule`, repository `vox-mcp`, workflow `publish.yml`, environment
`pypi`. Only the upload job requests an OIDC token; build and test jobs have no
publishing credentials. A retry checks PyPI for artifacts already uploaded;
do not move an existing tag or try to overwrite a published version.
