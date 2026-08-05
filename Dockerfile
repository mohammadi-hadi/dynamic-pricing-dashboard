# Dynamic Pricing Lab dashboard.
#   docker run --rm -p 8501:8501 ghcr.io/mohammadi-hadi/dynamic-pricing-dashboard
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/mohammadi-hadi/dynamic-pricing-dashboard" \
      org.opencontainers.image.description="Dynamic pricing simulator: demand learning, forward-looking customers, and advertising" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[dashboard]"

RUN useradd --create-home app
USER app
WORKDIR /home/app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["dynamic-pricing-lab", "--server.address=0.0.0.0", "--server.headless=true"]
