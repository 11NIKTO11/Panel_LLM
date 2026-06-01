try:
    from any_llm.provider import ProviderFactory
except Exception:
    # Optional dependency; provide a graceful fallback
    ProviderFactory = None


def check_response_api_support(provider_name: str) -> bool:
    if ProviderFactory is None:
        print("any-llm not installed; assuming no responses API support.")
        return False
    try:
        provider_class = ProviderFactory.get_provider_class(provider_name)
        return getattr(provider_class, 'SUPPORTS_RESPONSES', False)
    except Exception as e:
        print(f"Error checking provider {provider_name}: {e}")
        return False
