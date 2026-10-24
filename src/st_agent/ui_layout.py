"""Scoped presentation styles for the native Streamlit analysis workspace.

The pixel-height declaration remains in Python because Streamlit 1.63 uses it
to enable the native scroll/autoscroll surface. Desktop CSS changes its flex
basis, not its identity or event handlers. No browser-to-Python resize loop.
"""

from __future__ import annotations

import re


def conversation_layout_css(history_key: str) -> str:
    if re.fullmatch(r"chat_history_[A-Za-z0-9_-]+", history_key) is None:
        raise ValueError("Invalid native history container key")

    history = f".st-key-{history_key}"
    history_wrapper = (
        '.st-key-conversation_panel > [data-testid="stLayoutWrapper"]'
        f":has(> {history})"
    )
    # A missing/unavailable figure keeps the bounded, standalone chat layout.
    desktop = (
        '.st-key-analysis_workspace:has('
        '.st-key-figure_panel [data-testid="stPlotlyChart"])'
    )
    row = (
        desktop + ' > [data-testid="stLayoutWrapper"]'
        ' > [data-testid="stHorizontalBlock"]'
    )
    column = row + ' > [data-testid="stColumn"]:first-child'
    return f"""<style>
        {history_wrapper} {{
            height: clamp(260px, calc(100dvh - 330px), 480px) !important;
            flex: 0 0 auto;
            overflow: hidden;
        }}
        {history} {{
            height: 100% !important;
            min-height: 0;
            overscroll-behavior-y: contain;
        }}
        {history} [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        {history} [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {{
            font-size: 15px; line-height: 1.55;
        }}
        /* Match Streamlit 1.63's 640px column stacking breakpoint. */
        @media (min-width: 641px) {{
            {row} {{align-items: stretch;}}
            {column} {{display: flex; min-height: 0;}}
            {column} > [data-testid="stVerticalBlock"] {{
                height: auto !important;
                min-height: 0;
                flex: 1;
            }}
            {column} > [data-testid="stVerticalBlock"]
                > [data-testid="stLayoutWrapper"]:has(> .st-key-conversation_panel) {{
                height: auto !important;
                min-height: 0;
                flex: 1;
            }}
            {desktop} .st-key-conversation_panel {{
                min-height: 0;
                flex: 1;
            }}
            {desktop} .st-key-conversation_panel > * {{flex-shrink: 0;}}
            {desktop} {history_wrapper} {{
                height: 0 !important;
                min-height: 260px;
                flex: 1 1 0 !important;
            }}
        }}
        </style>"""
