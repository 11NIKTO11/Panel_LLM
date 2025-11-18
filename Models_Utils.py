from any_llm.provider import ProviderFactory

def check_response_api_support(provider_name: str) -> bool:
    try:
        provider_class = ProviderFactory.get_provider_class(provider_name)
        return provider_class.SUPPORTS_RESPONSES
    except Exception as e:
        print(f"Error checking provider {provider_name}: {e}")
        return False