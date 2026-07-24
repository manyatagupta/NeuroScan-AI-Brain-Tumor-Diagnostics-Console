import g4f

def test_g4f():
    providers = [g4f.Provider.Blackbox, g4f.Provider.DDG, g4f.Provider.ChatgptFree, None]
    for provider in providers:
        try:
            print(f"Testing provider: {provider}")
            response = g4f.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, write a 1 sentence medical summary."}],
                provider=provider
            )
            print(f"Success with {provider}: {response}")
            return
        except Exception as e:
            print(f"Failed with {provider}: {e}")

test_g4f()
