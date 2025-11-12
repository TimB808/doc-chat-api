#!/usr/bin/env bash
set -euo pipefail

FILE_ID="${FILE_ID:?set FILE_ID}"
QUESTION="${QUESTION:-Summarise the main themes of this document in under 50 words}"
ALPHAS="${ALPHAS:-0.6}"
LAMBDAS="${LAMBDAS:-1.0 0.6 0.3}"

export DEBUG_HYBRID="${DEBUG_HYBRID:-1}"

for A in $ALPHAS; do
  export HYBRID_ALPHA="$A"
  echo "==== α=$A ===="
  for L in $LAMBDAS; do
    export MMR_LAMBDA="$L"
    echo "-- λ=$L --"

    # Capture both response body and HTTP status code
    http_code=$(curl -s -w "%{http_code}" -o /tmp/sweep_response.json -X POST "http://localhost:8000/api/chat" \
      -H "Content-Type: application/json" \
      -d "{\"question\":\"$QUESTION\",\"file_id\":\"$FILE_ID\",\"alpha\":$A}")

    response=$(cat /tmp/sweep_response.json)
    rm -f /tmp/sweep_response.json

    if [ "$http_code" != "200" ]; then
      echo "Error: HTTP $http_code - $response" >&2
      continue
    fi

    if [ -z "$response" ] || [ "$response" = "null" ]; then
      echo "Error: Empty or null response from API" >&2
      continue
    fi

    # Try to parse with jq, handle errors gracefully
    parsed=$(echo "$response" | jq --arg lambda "$L" --arg alpha "$A" '
      .context as $ctx
      | $ctx // []
      | {
          lambda: $lambda,
          alpha:  $alpha,
          k:      length,
          scores: (map(.score) // [])
        }
      # Build lightweight “dup keys” by normalising text and taking a prefix
      | . + {
          dup_keys: (
            ($ctx // [])
            | map(
                (.text // "")
                | ascii_downcase
                | gsub("\\s+"; " ")             # collapse whitespace
                | gsub("[^a-z0-9 ]"; "")        # strip punctuation/symbols
                | .[0:120]                      # take a 120-char prefix
              )
          )
        }
      # duplicate_ratio = fraction of items that belong to a repeated prefix group
      | . + {
          duplicate_ratio:
            (if (.dup_keys | length) > 0
            then (
              .dup_keys
              | group_by(.)       # groups of same prefix
              | map(length)
              | map(select(. > 1))
              | add // 0
            ) / (.dup_keys | length)
            else 0 end)
        }
      # Optional: preview a couple of normalised prefixes to eyeball near-duplicates
      | . + {
          sample_prefixes: (.dup_keys[0:3] // [])
        }
      | del(.dup_keys)
    ')

    if [ $? -ne 0 ]; then
      echo "Error: Failed to parse API response - $response" >&2
      echo "jq error: $parsed" >&2
      continue
    fi

    echo "$parsed"
  done
done
