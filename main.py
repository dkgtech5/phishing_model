from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import tldextract

from predict_url import extract_features, model, EXACT_LEGITIMATE_DOMAINS, normalize_url

app = FastAPI(title="SafeGuard AI API")

class URLRequest(BaseModel):
    url: str

@app.get("/")
def home():
    return {"status": "SafeGuard AI Backend Running"}

@app.post("/predict")
async def predict_phishing(request: URLRequest):
    raw_input = request.url.strip()
    if not raw_input:
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    
    url = normalize_url(raw_input)

    try:
        ext = tldextract.extract(url)
        registered_domain = f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else ext.domain.lower()

        # Rule 1: Exact match for verified whitelist domains
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

        # Rule 2: Invalid Top-Level Domain
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

        # Rule 3: Brand spoofing / typo-squatting
        if any(b in ext.domain.lower() for b in ['esewa', 'khalti', 'paypal', 'daraz']) and registered_domain not in EXACT_LEGITIMATE_DOMAINS:
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

        # Rule 4: Machine Learning Model Extraction & Prediction
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

    except BaseException as e:
        print(f"[API Fallback Triggered] {url}: {str(e)}")
        
        is_https = url.startswith("https://")
        has_at_symbol = "@" in url
        has_suspicious_symbols = url.count('-') >= 3 or url.count('.') >= 4
        
        is_phish_heuristic = not is_https or has_at_symbol or has_suspicious_symbols
        status_label = "PHISHING" if is_phish_heuristic else "SAFE"
        risk = 80.0 if is_phish_heuristic else 15.0

        return {
            "url": url,
            "status": status_label,
            "risk_score": risk,
            "confidence_legitimate": round(100.0 - risk, 2),
            "confidence_phishing": risk,
            "security_checks": {
                "https_enabled": is_https,
                "trusted_domain": not is_phish_heuristic,
                "no_suspicious_redirect": not has_at_symbol,
                "clean_url_structure": not has_suspicious_symbols
            }
        }