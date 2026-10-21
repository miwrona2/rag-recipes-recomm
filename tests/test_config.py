from app.core.config import Settings


def test_settings_initialization():
    settings = Settings(
        PORT=8080,
        ENVIRONMENT="test",
        CORS_ORIGINS="http://localhost:3000,http://localhost:5173",
        OPENAI_API_KEY="test-key",
    )
    assert settings.PORT == 8080
    assert settings.ENVIRONMENT == "test"
    assert "http://localhost:3000" in settings.CORS_ORIGINS
    assert "http://localhost:5173" in settings.CORS_ORIGINS
    assert settings.OPENAI_API_KEY == "test-key"
