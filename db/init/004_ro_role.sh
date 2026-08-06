#!/bin/bash

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     -v ro_password="${CHATBOT_RO_PASSWORD:-chatbot_ro_dev_pw}" <<-'EOSQL'

    CREATE ROLE chatbot_ro LOGIN PASSWORD :'ro_password';

    -- Connect + read the public schema, nothing more.
    GRANT CONNECT ON DATABASE :"DBNAME" TO chatbot_ro;
    GRANT USAGE   ON SCHEMA public      TO chatbot_ro;

    GRANT SELECT ON ALL TABLES IN SCHEMA public TO chatbot_ro;

    -- Application-owned tables are NOT agent-readable. routing_decisions and
    -- user_memory contain conversation content; letting a prompt-injected
    -- query SELECT from them would turn the SQL tool into an exfiltration
    -- channel.
    REVOKE ALL ON routing_decisions FROM chatbot_ro;
    REVOKE ALL ON user_memory       FROM chatbot_ro;

    -- Deliberately NO `ALTER DEFAULT PRIVILEGES`: tables created later at
    -- runtime (e.g. the LangGraph checkpointer's tables) must not become
    -- readable by the agent role by default.

    -- Belt and braces: even a SELECT-only role could hold open a transaction
    -- or run an expensive scan. Force read-only transactions and hard limits.
    ALTER ROLE chatbot_ro SET default_transaction_read_only        = on;
    ALTER ROLE chatbot_ro SET statement_timeout                    = '5s';
    ALTER ROLE chatbot_ro SET idle_in_transaction_session_timeout  = '10s';
    ALTER ROLE chatbot_ro SET lock_timeout                         = '2s';

EOSQL

echo "004_ro_role.sh: chatbot_ro created (read-only, 5s statement timeout)"
