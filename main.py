from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import tldextract

# Import model and feature extraction engine from predict_url.py
from predict_url import extract_features, model, EXACT_LEGITIMATE_DOMAINS

app = FastAPI(title="SafeGuard AI API")

class URLRequest(BaseModel):
    url: str

@app.get("/")
def home():
    return {"status": "SafeGuard AI Backend Running"}

@app.post("/predict")
async def predict_phishing(request: URLRequest):
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    
    try:
        ext = tldextract.extract(url)
        registered_domain = f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else ext.domain.lower()

        # Rule 1: Exact match for verified legitimate domains
        if registered_domain in EXACT_LEGITIMATE_DOMAINS and ext.suffix and not ext.subdomain:
            return {
                "url": url,
                "status": "SAFE",
                "risk_score": 0.0,
                "confidence_legitimate": 100.0,
                "confidence_phishing": 0.0,
                "security_checks": {
                    "https_enabled": url.startswith("https://"),
                    "trusted_domain": True,
                    "no_suspicious_redirect": True,
                    "clean_url_structure": True
                }
            }

        # Rule 2: Invalid Top-Level Domain (e.g., google.com.a)
        if not bool(ext.suffix):
            return {
                "url": url,
                "status": "PHISHING",
                "risk_score": 100.0,
                "confidence_legitimate": 0.0,
                "confidence_phishing": 100.0,
                "security_checks": {
                    "https_enabled": url.startswith("https://"),
                    "trusted_domain": False,
                    "no_suspicious_redirect": False,
                    "clean_url_structure": False
                }
            }

        # Rule 3: Brand spoofing / typo-squatting (e.g., esewadkg.com.np)
        if any(b in ext.domain.lower() for b in ['esewa', 'khalti', 'paypal']) and registered_domain not in EXACT_LEGITIMATE_DOMAINS:
            return {
                "url": url,
                "status": "PHISHING",
                "risk_score": 100.0,
                "confidence_legitimate": 0.0,
                "confidence_phishing": 100.0,
                "security_checks": {
                    "https_enabled": url.startswith("https://"),
                    "trusted_domain": False,
                    "no_suspicious_redirect": False,
                    "clean_url_structure": False
                }
            }

        # Rule 4: Machine Learning Model Extraction & Prediction (Async Offload)
        df_features = await run_in_threadpool(extract_features, url)
        pred = model.predict(df_features)[0]
        prob = model.predict_proba(df_features)[0]

        conf_legit = round(float(prob[0]) * 100, 2)
        conf_phish = round(float(prob[1]) * 100, 2)
        label = "PHISHING" if pred == 1 else "SAFE"

        return {
            "url": url,
            "status": label,
            "risk_score": conf_phish,
            "confidence_legitimate": conf_legit,
            "confidence_phishing": conf_phish,
            "security_checks": {
                "https_enabled": url.startswith("https://"),
                "trusted_domain": pred == 0,
                "no_suspicious_redirect": not ("redirect=" in url or "@" in url),
                "clean_url_structure": url.count('-') < 2 and url.count('.') < 4
            }
        }

    except Exception as e:
        print(f"[API Error] Failed to analyze URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")