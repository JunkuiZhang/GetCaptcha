import requests
import re
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.encoders import encode_base64
import smtplib

from_addr = "junkuizhang@126.com"
to_addr = "junkuizhang@126.com"
pwd = "ZJKzhangjunkui01"
smtp_url = "smtp.126.com"
while True:
	url = "http://210.42.121.241/servlet/GenImg"
	session = requests.session()
	res = session.get(url)
	f = open("./get_captcha/0.jpg", "wb")
	f.write(res.content)
	f.close()

	captcha = input("Enter the strings: ")
	post_data = {
		"id": "2013301000021",
		"pwd": "zjk1995",
		"xdvfb": captcha
	}

	cookie = re.findall('kie (.*) for', str(res.cookies))[0]
	headers = {
		"Cookie": cookie
	}

	login_url = "http://210.42.121.241/servlet/Login"
	res = session.post(login_url, headers=headers, data=post_data)
	if re.findall('验证码错误', res.text):
		print("Invalid captcha input!")
	else:
		server = smtplib.SMTP(smtp_url, 25)
		server.login(from_addr, pwd)

		msg = MIMEMultipart()
		message = MIMEText(captcha, "plain", "utf-8")

		def get_format_addr(s):
			name, addr = parseaddr(s)
			return formataddr((Header(name, "utf-8").encode(), addr))

		msg["From"] = get_format_addr("Junkui Zhang <%s>" % from_addr)
		msg["To"] = get_format_addr("Me <%s>" % to_addr)
		msg["Subject"] = Header("<!CAPTCHA!>", "utf-8").encode()
		msg.attach(message)

		with open("./get_captcha/test.jpg", "rb") as f:
			pic = MIMEBase("image", "jpg", filename="0.jpg")
			pic.add_header("Content-Disposition", "attachment", filename="0.jpg")
			pic.add_header("Content-ID", "<0>")
			pic.set_payload(f.read())
			encode_base64(pic)
			msg.attach(pic)

		server.sendmail(from_addr, [to_addr], msg.as_bytes())
		server.close()
