#!/usr/bin/env python3
"""
Cortex Analyst Chat — Flask app that embeds as an iframe in Atlan.

Flow:
1. Atlan sends asset GUID via postMessage (ATLAN_AUTH_CONTEXT)
2. App reads "Cortex Analyst Details" custom metadata from Atlan to get the
   semantic view's fully qualified name
3. User asks questions in natural language
4. App proxies them to Snowflake Cortex Analyst REST API
5. Returns generated SQL + text response
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
import time
import requests as http_requests
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
import snowflake.connector
from cryptography.hazmat.primitives import serialization

load_dotenv()

app = Flask(__name__)

CORS(app, origins="*")

@app.after_request
def allow_iframe_embedding(response):
    """Allow Atlan to embed this app in an iframe."""
    response.headers.pop("X-Frame-Options", None)
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    return response

# ── Configuration ─────────────────────────────────────────────────────
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER", "")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE", "")
SNOWFLAKE_AUTHENTICATOR = os.getenv("SNOWFLAKE_AUTHENTICATOR", "snowflake")
SNOWFLAKE_PRIVATE_KEY = os.getenv("SNOWFLAKE_PRIVATE_KEY", "")
SNOWFLAKE_PRIVATE_KEY_PASSPHRASE = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")
SNOWFLAKE_TOKEN = os.getenv("SNOWFLAKE_TOKEN", "")  # Programmatic Access Token
ATLAN_HOST = os.getenv("ATLAN_HOST", "https://partner-sandbox.atlan.com")
ATLAN_API_TOKEN = os.getenv("ATLAN_API_TOKEN", "")


# ── Snowflake Session via connector (supports SSO) ───────────────────
class SnowflakeSession:
    """
    Uses snowflake-connector-python for auth (supports externalbrowser SSO),
    then extracts the session token for Cortex Analyst REST API calls.
    """

    def __init__(self):
        self.account = SNOWFLAKE_ACCOUNT
        # Build the base URL for REST API calls.
        # Underscores are not valid DNS chars — replace with hyphens for the hostname only.
        # The connector's account= param keeps the underscore; the URL needs hyphens.
        account_hostname = SNOWFLAKE_ACCOUNT.replace("_", "-").lower()
        self.base_url = f"https://{account_hostname}.snowflakecomputing.com"
        self._conn = None
        self._token = None
        self._token_expiry = 0

    def _connect(self):
        """Establish a Snowflake connection (SSO, password, or key-pair)."""
        connect_args = dict(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            warehouse=SNOWFLAKE_WAREHOUSE,
            role=SNOWFLAKE_ROLE,
        )

        if SNOWFLAKE_TOKEN and SNOWFLAKE_TOKEN.strip():
            print("  Using Programmatic Access Token (PAT)...")
            connect_args["authenticator"] = "programmatic_access_token"
            connect_args["token"] = SNOWFLAKE_TOKEN.replace("\\n", "").replace("\n", "").strip()
        elif SNOWFLAKE_AUTHENTICATOR == "externalbrowser":
            print("  Opening browser for Snowflake SSO login...")
            connect_args["authenticator"] = "externalbrowser"
        elif SNOWFLAKE_PRIVATE_KEY and SNOWFLAKE_PRIVATE_KEY.strip():
            print("  Using key-pair authentication...")
            passphrase = (
                SNOWFLAKE_PRIVATE_KEY_PASSPHRASE.encode()
                if SNOWFLAKE_PRIVATE_KEY_PASSPHRASE
                else None
            )
            pem = SNOWFLAKE_PRIVATE_KEY.replace("\\n", "\n")
            p_key = serialization.load_pem_private_key(
                pem.encode(),
                password=passphrase,
            )
            pkb = p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            connect_args["private_key"] = pkb
        else:
            connect_args["password"] = SNOWFLAKE_PASSWORD

        self._conn = snowflake.connector.connect(**connect_args)

        # Extract session token from the connection's REST object
        rest = self._conn.rest
        self._token = rest.token
        self._master_token = rest.master_token
        self._token_expiry = time.time() + 3600  # Refresh after 1 hour

        print(f"  Snowflake connected! Token obtained.")

    def get_token(self) -> str:
        """Return a valid session token, reconnecting if needed."""
        if not self._token or time.time() >= self._token_expiry:
            self._connect()
        return self._token

    def get_connection(self):
        """Return the raw snowflake connection for SQL queries."""
        if not self._conn or self._conn.is_closed():
            self._connect()
        return self._conn


# ── Cortex Analyst Client ────────────────────────────────────────────
class CortexAnalystClient:
    """Sends natural language questions to Snowflake Cortex Analyst."""

    def __init__(self, session: SnowflakeSession):
        self.session = session
        self.api_url = f"{session.base_url}/api/v2/cortex/analyst/message"
        print(f"  Cortex Analyst endpoint: {self.api_url}")

    def ask(self, semantic_view_fqn: str, question: str,
            history: list = None) -> Dict[str, Any]:
        """
        Send a question to Cortex Analyst.

        Args:
            semantic_view_fqn: e.g. "WIDE_WORLD_IMPORTERS.PROCESSED_GOLD.SALES_DOMAIN_SIMPLE"
            question: natural language question
            history: prior messages for multi-turn conversation

        Returns:
            dict with 'text', 'sql', 'suggestions' keys
        """
        messages = []

        if history:
            messages.extend(history)

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": question}],
        })

        # Cortex Analyst REST API: use top-level "semantic_view" key for Semantic View objects.
        # (Not nested inside "semantic_model" — that key is for inline YAML strings.)
        payload = {
            "messages": messages,
            "semantic_view": semantic_view_fqn,
        }

        headers = {
            "Authorization": f'Snowflake Token="{self.session.get_token()}"',
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        resp = http_requests.post(
            self.api_url, json=payload, headers=headers, timeout=60
        )

        if not resp.ok:
            # Log the full response for debugging
            print(f"  [Cortex Analyst] HTTP {resp.status_code}: {resp.text[:500]}")

        resp.raise_for_status()
        data = resp.json()

        result = {"text": None, "sql": None, "suggestions": [], "data": None}

        for block in data.get("message", {}).get("content", []):
            if block["type"] == "text":
                result["text"] = block["text"]
            elif block["type"] == "sql":
                result["sql"] = block.get("statement", block.get("text", ""))
            elif block["type"] == "suggestions":
                result["suggestions"] = block.get("suggestions", [])

        # Execute the SQL and return results
        if result["sql"]:
            try:
                conn = self.session.get_connection()
                cur = conn.cursor()
                cur.execute(result["sql"])
                columns = [col[0] for col in cur.description]
                rows = cur.fetchmany(100)  # Limit to 100 rows
                result["data"] = {
                    "columns": columns,
                    "rows": [list(row) for row in rows],
                    "row_count": cur.rowcount,
                }
                cur.close()
            except Exception as e:
                result["data_error"] = str(e)

        # Return the assistant message for conversation history
        result["assistant_message"] = data.get("message", {})

        return result


# ── Initialize clients ────────────────────────────────────────────────
sf_session = None
cortex_client = None

if SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER:
    print("Initializing Snowflake session...")
    sf_session = SnowflakeSession()
    if SNOWFLAKE_AUTHENTICATOR == "externalbrowser":
        # SSO: connect eagerly so the browser opens at startup (local dev only)
        sf_session.get_token()
        cortex_client = CortexAnalystClient(sf_session)
        print("Cortex Analyst client ready!")
    else:
        # Password/key-pair: connect lazily on first request (production)
        try:
            sf_session.get_token()
            cortex_client = CortexAnalystClient(sf_session)
            print("Cortex Analyst client ready!")
        except Exception as e:
            print(f"Warning: Snowflake connection deferred — {e}")
            cortex_client = CortexAnalystClient(sf_session)
            print("Cortex Analyst client created (will connect on first request).")


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/space/<space_guid>")
def get_space_info(space_guid):
    """Look up an Atlan asset by GUID via Atlan REST API, read custom metadata."""
    if not ATLAN_API_TOKEN:
        return jsonify({"success": False, "error": "Atlan API not configured."})

    try:
        # ── Step 1: Resolve the CM typedef to get attribute ID mappings ──
        # Atlan stores custom metadata under business-metadata typedefs.
        # We need to find the attribute IDs for "Cortex Analyst Details".
        typedef_resp = http_requests.get(
            f"{ATLAN_HOST}/api/meta/types/typedefs?type=business_metadata",
            headers={
                "Authorization": f"Bearer {ATLAN_API_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        typedef_resp.raise_for_status()

        # Build a map: attribute display name -> attribute internal name
        cm_attr_map = {}  # internal_name -> display_name
        cm_typedef_name = None
        for td in typedef_resp.json().get("businessMetadataDefs", []):
            if td.get("displayName") == "Cortex Analyst Details":
                cm_typedef_name = td["name"]
                for attr in td.get("attributeDefs", []):
                    cm_attr_map[attr["name"]] = attr.get("displayName", attr["name"])
                break

        # ── Step 2: Get the asset by GUID ────────────────────────────────
        asset_resp = http_requests.get(
            f"{ATLAN_HOST}/api/meta/entity/guid/{space_guid}",
            params={"minExtInfo": "false", "ignoreRelationships": "true"},
            headers={
                "Authorization": f"Bearer {ATLAN_API_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        asset_resp.raise_for_status()
        entity = asset_resp.json().get("entity", {})
        attrs = entity.get("attributes", {})
        biz_attrs = entity.get("businessAttributes", {})

        # ── Step 3: Extract Cortex Analyst Details ───────────────────────
        cortex_raw = biz_attrs.get(cm_typedef_name, {}) if cm_typedef_name else {}

        if not cortex_raw:
            return jsonify({
                "success": False,
                "error": "No 'Cortex Analyst Details' custom metadata found.",
                "debug": {"asset_name": attrs.get("name")},
            })

        # Remap internal attribute names to display names
        cortex_meta = {}
        for internal_name, value in cortex_raw.items():
            display = cm_attr_map.get(internal_name, internal_name)
            cortex_meta[display] = value

        fqn = cortex_meta.get("fullyQualifiedName", "")
        view_name = cortex_meta.get("semanticViewName", attrs.get("name", ""))
        db = cortex_meta.get("database", "")
        schema = cortex_meta.get("schema", "")
        table_count = cortex_meta.get("tableCount", 0)
        metric_count = cortex_meta.get("metricCount", 0)
        dim_count = cortex_meta.get("dimensionCount", 0)

        acct_path = SNOWFLAKE_ACCOUNT.lower().replace("-", "/")
        sf_url = (
            f"https://app.snowflake.com/{acct_path}/"
            f"#/cortex/analyst/databases/{db}/schemas/{schema}/"
            f"semanticView/{view_name}/edit"
        )

        description = (
            attrs.get("description")
            or attrs.get("userDescription")
            or f"Cortex Analyst semantic view with {table_count} tables, "
               f"{dim_count} dimensions, {metric_count} metrics"
        )

        return jsonify({
            "success": True,
            "semantic_view_fqn": fqn,
            "name": view_name,
            "description": description,
            "database": db,
            "schema": schema,
            "table_count": table_count,
            "metric_count": metric_count,
            "dimension_count": dim_count,
            "snowflake_url": sf_url,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/views")
def list_views():
    """List all available semantic views (for standalone/test mode)."""
    views = [
        {"name": "SALES_DOMAIN_SIMPLE", "fqn": "WIDE_WORLD_IMPORTERS.PROCESSED_GOLD.SALES_DOMAIN_SIMPLE",
         "description": "Basic sales analytics", "tables": 5, "dims": 24, "metrics": 0},
        {"name": "SALES_DOMAIN_ATLAN_ENHANCED", "fqn": "WIDE_WORLD_IMPORTERS.PROCESSED_GOLD.SALES_DOMAIN_ATLAN_ENHANCED",
         "description": "Enriched sales analytics with Atlan metadata", "tables": 4, "dims": 26, "metrics": 7},
        {"name": "WWI_SALES_SEMANTIC_VIEW", "fqn": "WIDE_WORLD_IMPORTERS.PROCESSED_GOLD.WWI_SALES_SEMANTIC_VIEW",
         "description": "WWI sales semantic view", "tables": 4, "dims": 19, "metrics": 5},
        {"name": "CMP_ANALYST", "fqn": "WIDE_WORLD_IMPORTERS.PROCESSED_GOLD.CMP_ANALYST",
         "description": "CMP analyst view", "tables": 1, "dims": 17, "metrics": 0},
        {"name": "CMP_PLAIN", "fqn": "WIDE_WORLD_IMPORTERS.PROCESSED_GOLD.CMP_PLAIN",
         "description": "CMP plain view", "tables": 6, "dims": 248, "metrics": 0},
        {"name": "CMP_RAW", "fqn": "WIDE_WORLD_IMPORTERS.PROCESSED_GOLD.CMP_RAW",
         "description": "CMP raw view", "tables": 6, "dims": 248, "metrics": 0},
    ]
    return jsonify({"success": True, "views": views})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Proxy a question to Cortex Analyst and return the response."""
    if not cortex_client:
        return jsonify({
            "success": False,
            "error": "Cortex Analyst not configured. Check Snowflake credentials.",
        }), 503

    data = request.json
    semantic_view_fqn = data.get("semantic_view_fqn")
    message = data.get("message")
    history = data.get("history", [])

    if not semantic_view_fqn or not message:
        return jsonify({"success": False, "error": "Missing semantic_view_fqn or message"}), 400

    try:
        result = cortex_client.ask(
            semantic_view_fqn=semantic_view_fqn,
            question=message,
            history=history,
        )

        return jsonify({
            "success": True,
            "response": result.get("text", "Query processed successfully"),
            "sql": result.get("sql"),
            "data": result.get("data"),
            "data_error": result.get("data_error"),
            "suggestions": result.get("suggestions", []),
            "assistant_message": result.get("assistant_message"),
        })

    except http_requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else 500
        body = ""
        try:
            body = e.response.json().get("message", str(e))
        except Exception:
            body = str(e)

        # If 401, try to refresh the token
        if status == 401:
            try:
                sf_session.get_token()
                # Retry once
                result = cortex_client.ask(
                    semantic_view_fqn=semantic_view_fqn,
                    question=message,
                    history=history,
                )
                return jsonify({
                    "success": True,
                    "response": result.get("text"),
                    "sql": result.get("sql"),
                    "data": result.get("data"),
                    "suggestions": result.get("suggestions", []),
                    "assistant_message": result.get("assistant_message"),
                })
            except Exception:
                pass

        return jsonify({"success": False, "error": f"Cortex Analyst error ({status}): {body}"}), 502

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/config")
def get_config():
    return jsonify({
        "configured": bool(cortex_client),
        "account": SNOWFLAKE_ACCOUNT[:15] + "..." if SNOWFLAKE_ACCOUNT else None,
        "authenticator": SNOWFLAKE_AUTHENTICATOR,
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "app": "cortex-analyst-chat",
        "snowflake_connected": bool(cortex_client),
        "timestamp": datetime.now().isoformat(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
