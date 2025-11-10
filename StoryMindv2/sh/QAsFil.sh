python QAsFil.py --openai_model gpt-4.1-2025-04-14 \
                 --openai_key "Replace with your GPT API key" \
                 --openai_proxy "Replace with your GPT proxy address" \
                 --gemini_model gemini-2.0-flash \
                 --gemini_key "Replace with your Gemini API key" \
                 --gemini_proxy "Replace with your Gemini proxy address" \
                 --claude_model claude-3-7-sonnet-20250219 \
                 --claude_key "Replace with your Claude API key" \
                 --claude_proxy "Replace with your Claude proxy address" \
                 --vid_dir Friends \
                 --start 0 \
                 --end 999

                   
python export.py --vid_dir Friends --output_path json/filter_QAs.json