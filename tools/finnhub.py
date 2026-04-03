import os
import finnhub #pip3 install finnhub-python
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Initialize the Finnhub client
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

@tool
def get_stock_price(query: str) -> str:
    """
    Fetch the real-time price for a company name or a stock symbol.
    Example queries: 'Apple', 'Tesla', 'NVDA', 'Bitcoin'.
    """
    try:
        search_results = finnhub_client.symbol_lookup(query)
        if not search_results['result']:
            return f"Could not find any company or symbol matching '{query}'."

        # FIX: Add here to get the first dictionary in the list
        best_match = search_results['result'][0]

        # Now this will work because best_match is a dictionary
        symbol = best_match['symbol']
        company_name = best_match['description']

        # 2. Fetch the actual price using the found symbol
        quote = finnhub_client.quote(symbol)
        
        if quote['c'] == 0:
            return f"Found '{company_name}' ({symbol}), but no price data is currently available."
            
        return (f"Company: {company_name} ({symbol})\n"
                f"Current Price: ${quote['c']}\n"
                f"Change: {quote['d']} ({quote['dp']}%)\n"
                f"High: ${quote['h']}, Low: ${quote['l']}")

    except Exception as e:
        return f"Error processing stock request for '{query}': {str(e)}"
    
response = get_stock_price.invoke("Tesla")
print(response)