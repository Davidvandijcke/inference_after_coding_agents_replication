#!/bin/bash
# The agent's shell in the Claude route: the CLI holds the login token, its commands do not.
unset CLAUDE_CODE_OAUTH_TOKEN ANTHROPIC_API_KEY
exec /bin/bash "$@"
