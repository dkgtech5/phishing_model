import sys
import re
import datetime
import socket
import joblib
import pandas as pd
import numpy as np
import tldextract
from urllib.parse import urlparse

# Load trained pipeline and feature names
model = joblib.load('mlp_phishing_pipeline.pkl')
feature_names = joblib.load('feature_names.pkl')

# Top legitimate domains whitelist (Exact Matches Only)
EXACT_LEGITIMATE_DOMAINS = {
    'google.com', 'esewa.com.np', 'khalti.com', 
    'github.com', 'amazon.com', 'facebook.com', 
    'paypal.com', 'microsoft.com', 'apple.com',
    'flipkart.com', 'wikipedia.org', 'stackoverflow.com',
    'ncit.edu.np'
}

def normalize_url(url):
    """Ensures input always has a scheme for safe parsing."""
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url

def get_clean_domain(domain_str):
    domain_str = domain_str.split(':')[0]
    parts = domain_str.lower().split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return domain_str

def get_live_domain_info(raw_domain):
    clean_domain = get_clean_domain(raw_domain)
    
    # Defaults prevent model penalties on lookup timeouts, blocks, or failures
    data = {
        'time_domain_activation': 3650,  # 10 years default fallback
        'time_domain_expiration': 365,
        'qty_nameservers': 2,
        'qty_mx_servers': 1,
        'ttl_hostname': 300,
        'qty_ip_resolved': 1,
        'domain_spf': 1
    }

    # Global hard timeout for sockets
    socket.setdefaulttimeout(0.5)

    # 1. Non-blocking DNS Check
    try:
        import dns.resolver
        resolver = dns.resolver.Resolver()
        resolver.lifetime = 0.5
        resolver.timeout = 0.5

        try:
            a_answers = resolver.resolve(raw_domain, 'A')
            data['qty_ip_resolved'] = len(a_answers)
            data['ttl_hostname'] = getattr(a_answers, 'ttl', 300)
        except BaseException:
            data['qty_ip_resolved'] = 0

        try:
            mx_answers = resolver.resolve(clean_domain, 'MX')
            data['qty_mx_servers'] = len(mx_answers)
        except BaseException:
            data['qty_mx_servers'] = 0
    except BaseException:
        pass

    # 2. Crash-Proof WHOIS Check
    try:
        import whois
        w = whois.whois(clean_domain)
        
        if w:
            creation_date = getattr(w, 'creation_date', None)
            if isinstance(creation_date, list) and creation_date:
                creation_date = creation_date[0]
                
            expiration_date = getattr(w, 'expiration_date', None)
            if isinstance(expiration_date, list) and expiration_date:
                expiration_date = expiration_date[0]

            if isinstance(creation_date, datetime.datetime):
                data['time_domain_activation'] = max((datetime.datetime.now() - creation_date).days, 0)

            if isinstance(expiration_date, datetime.datetime):
                data['time_domain_expiration'] = max((expiration_date - datetime.datetime.now()).days, 0)

            ns = getattr(w, 'name_servers', None)
            if ns:
                data['qty_nameservers'] = len(ns) if isinstance(ns, list) else 1
    except BaseException:
        pass

    return data

def extract_features(url):
    url = normalize_url(url)
    ext = tldextract.extract(url)
    registered_domain = f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else ext.domain.lower()
    
    parsed = urlparse(url)
    netloc = parsed.netloc or parsed.path.split('/')[0]
    domain = netloc.split(':')[0].lower()
    path = parsed.path if parsed.netloc else ''
    params = parsed.query

    feats = {}
    
    # Base URL counts
    feats['qty_dot_url'] = url.count('.')
    feats['qty_hyphen_url'] = url.count('-')
    feats['qty_underline_url'] = url.count('_')
    feats['qty_slash_url'] = url.count('/')
    feats['qty_questionmark_url'] = url.count('?')
    feats['qty_equal_url'] = url.count('=')
    feats['qty_at_url'] = url.count('@')
    feats['qty_and_url'] = url.count('&')
    feats['qty_exclamation_url'] = url.count('!')
    feats['qty_space_url'] = url.count(' ')
    feats['qty_tilde_url'] = url.count('~')
    feats['qty_comma_url'] = url.count(',')
    feats['qty_plus_url'] = url.count('+')
    feats['qty_asterisk_url'] = url.count('*')
    feats['qty_hashtag_url'] = url.count('#')
    feats['qty_dollar_url'] = url.count('$')
    feats['qty_percent_url'] = url.count('%')
    feats['qty_tld_url'] = len(domain.split('.')) - 1 if '.' in domain else 0
    feats['length_url'] = len(url)

    # Domain Counts
    feats['qty_dot_domain'] = domain.count('.')
    feats['qty_hyphen_domain'] = domain.count('-')
    feats['qty_underline_domain'] = domain.count('_')
    feats['qty_slash_domain'] = domain.count('/')
    feats['qty_questionmark_domain'] = domain.count('?')
    feats['qty_equal_domain'] = domain.count('=')
    feats['qty_at_domain'] = domain.count('@')
    feats['qty_and_domain'] = domain.count('&')
    feats['qty_exclamation_domain'] = domain.count('!')
    feats['qty_space_domain'] = domain.count(' ')
    feats['qty_tilde_domain'] = domain.count('~')
    feats['qty_comma_domain'] = domain.count(',')
    feats['qty_plus_domain'] = domain.count('+')
    feats['qty_asterisk_domain'] = domain.count('*')
    feats['qty_hashtag_domain'] = domain.count('#')
    feats['qty_dollar_domain'] = domain.count('$')
    feats['qty_percent_domain'] = domain.count('%')
    feats['qty_vowels_domain'] = len(re.findall(r'[aeiouAEIOU]', domain))
    feats['domain_length'] = len(domain)
    feats['domain_in_ip'] = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else 0
    feats['server_client_domain'] = 1 if 'server' in domain or 'client' in domain else 0

    # Directory Counts
    feats['qty_dot_directory'] = path.count('.')
    feats['qty_hyphen_directory'] = path.count('-')
    feats['qty_underline_directory'] = path.count('_')
    feats['qty_slash_directory'] = path.count('/')
    feats['qty_questionmark_directory'] = path.count('?')
    feats['qty_equal_directory'] = path.count('=')
    feats['qty_at_directory'] = path.count('@')
    feats['qty_and_directory'] = path.count('&')
    feats['qty_exclamation_directory'] = path.count('!')
    feats['qty_space_directory'] = path.count(' ')
    feats['qty_tilde_directory'] = path.count('~')
    feats['qty_comma_directory'] = path.count(',')
    feats['qty_plus_directory'] = path.count('+')
    feats['qty_asterisk_directory'] = path.count('*')
    feats['qty_hashtag_directory'] = path.count('#')
    feats['qty_dollar_directory'] = path.count('$')
    feats['qty_percent_directory'] = path.count('%')
    feats['directory_length'] = len(path)

    # Params Counts
    feats['qty_dot_params'] = params.count('.')
    feats['qty_hyphen_params'] = params.count('-')
    feats['qty_underline_params'] = params.count('_')
    feats['qty_slash_params'] = params.count('/')
    feats['qty_questionmark_params'] = params.count('?')
    feats['qty_equal_params'] = params.count('=')
    feats['qty_at_params'] = params.count('@')
    feats['qty_and_params'] = params.count('&')
    feats['qty_exclamation_params'] = params.count('!')
    feats['qty_space_params'] = params.count(' ')
    feats['qty_tilde_params'] = params.count('~')
    feats['qty_comma_params'] = params.count(',')
    feats['qty_plus_params'] = params.count('+')
    feats['qty_asterisk_params'] = params.count('*')
    feats['qty_hashtag_params'] = params.count('#')
    feats['qty_dollar_params'] = params.count('$')
    feats['qty_percent_params'] = params.count('%')
    feats['params_length'] = len(params)
    feats['qty_params'] = len(params.split('&')) if params else 0

    # Safe network domain fetch
    try:
        live_data = get_live_domain_info(domain)
        feats.update(live_data)
    except BaseException:
        pass

    feats['tls_ssl_certificate'] = 1 if url.startswith('https') else 0
    feats['url_shortened'] = 1 if len(domain) < 10 and any(s in domain for s in ['bit.ly', 'goo.gl', 't.co', 'tinyurl']) else 0
    feats['url_google_index'] = 1
    feats['domain_google_index'] = 1
    feats['asn_ip'] = 0
    feats['email_in_url'] = 1 if re.search(r'[\w\.-]+@[\w\.-]+', url) else 0

    row = {f: feats.get(f, -1) for f in feature_names}
    return pd.DataFrame([row])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict_url.py <URL>")
        sys.exit(1)

    url_input = normalize_url(sys.argv[1])
    ext = tldextract.extract(url_input)
    registered_domain = f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else ext.domain.lower()

    if registered_domain in EXACT_LEGITIMATE_DOMAINS and ext.suffix and not ext.subdomain:
        label = "LEGITIMATE"
        prob = [100.0, 0.0]
    elif not bool(ext.suffix):
        label = "PHISHING"
        prob = [0.0, 100.0]
    elif any(b in ext.domain.lower() for b in ['esewa', 'khalti', 'paypal']) and registered_domain not in EXACT_LEGITIMATE_DOMAINS:
        label = "PHISHING"
        prob = [0.0, 100.0]
    else:
        df_features = extract_features(url_input)
        pred = model.predict(df_features)[0]
        prob_raw = model.predict_proba(df_features)[0]
        
        label = "PHISHING" if pred == 1 else "LEGITIMATE"
        prob = [prob_raw[0] * 100, prob_raw[1] * 100]

    print(f"\nTarget URL: {url_input}")
    print(f"Prediction: {label}")
    print(f"Confidence (Legitimate vs Phishing): {prob[0]:.2f}% vs {prob[1]:.2f}%\n")