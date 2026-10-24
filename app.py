"""Launch with: streamlit run app.py"""

if __name__ == "__main__":
    # Windows spawn re-imports this file as __mp_main__; workers must not start UI.
    from st_agent.streamlit_app import render_app

    render_app()
