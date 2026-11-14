"""Source credibility database (0-100). Extend freely."""
CRED = {
 # high
 "reuters.com":95,"apnews.com":95,"bbc.com":92,"bbc.co.uk":92,"npr.org":90,"pbs.org":90,
 "nytimes.com":88,"theguardian.com":85,"washingtonpost.com":85,"wsj.com":88,"bloomberg.com":88,
 "economist.com":90,"ft.com":90,"aljazeera.com":80,"dw.com":88,"france24.com":85,
 "thehindu.com":85,"indianexpress.com":80,"hindustantimes.com":72,"timesofindia.indiatimes.com":68,
 "ndtv.com":72,"livemint.com":78,"scroll.in":72,"theprint.in":75,"deccanherald.com":75,
 "nature.com":96,"science.org":96,"who.int":95,"cdc.gov":94,"nasa.gov":95,
 # medium
 "cnn.com":70,"nbcnews.com":78,"abcnews.go.com":78,"cbsnews.com":78,"usatoday.com":72,
 "foxnews.com":55,"nypost.com":50,"dailymail.co.uk":42,"huffpost.com":60,"buzzfeed.com":50,
 # low / unreliable / satire
 "infowars.com":5,"naturalnews.com":8,"beforeitsnews.com":6,"breitbart.com":30,
 "zerohedge.com":25,"thegatewaypundit.com":10,"theonion.com":3,"worldnewsdailyreport.com":2,
 "yournewswire.com":3,"newspunch.com":5,"dailywire.com":35,
}
SATIRE = {"theonion.com","worldnewsdailyreport.com","babylonbee.com","thebeaverton.com"}
