import poplib
import email
from email.header import Header
from email.parser import Parser
from email.header import decode_header
from email.utils import parseaddr
import os
import csv

USER = "junkuizhang@126.com"
PWD = "ZJKzhangjunkui01"


class GetCaptchaData:

	def __init__(self):
		self.user = USER
		self.pwd = PWD
		self.server = "pop.126.com"
		self.str = ""

	def main(self):
		server = poplib.POP3(self.server)
		server.user(self.user)
		server.pass_(self.pwd)
		resp, mails, octets = server.list()

		def decode_str(s):
			value, charset = decode_header(s)[0]
			if charset:
				value = value.decode(charset)
			return value

		def guess_charset(msg):
			charset = msg.get_charset()
			if charset is None:
				content_type = msg.get("Content-Type", "").lower()
				pos = content_type.find("charset=")
				if pos >= 0:
					charset = content_type[pos + 8:].strip()
			return charset

		def print_info(msg, indent=0):
			if indent == 0:
				for header in ["From", "To", "Subject"]:
					value = msg.get(header, "")
					if value:
						if header == "Subject":
							value = decode_str(value)
						else:
							hdr, addr = parseaddr(value)
							name = decode_str(hdr)
							value = u'%s <%s>' % (name, addr)
					print("%s%s: %s" % ('  ' * indent, header, value))
			if msg.is_multipart():
				parts = msg.get_payload()
				for n, part in enumerate(parts):
					print("%spart %s" % ('  ' * indent, n))
					print("%s-------------------" % ("  " * indent))
					print_info(part, indent + 1)
			else:
				content_type = msg.get_content_type()
				if content_type == "text/plain" or content_type == "text/html":
					content = msg.get_payload(decode=True)
					charset = guess_charset(msg)
					if charset:
						content = content.decode(charset)
					print("%sText: %s" % ("  "* indent, content + "..."))
					self.str = content
				else:
					print("%sAttachment: %s" % ("  " * indent, content_type))
		if not os.path.exists("./Data"):
			os.mkdir("./Data")
		num = 0
		for index in range(1, len(mails) + 1):
			resp, lines, octets = server.retr(index)
			# msg_content = email.message_from_bytes(b"\n".join(lines))
			msg_content = b'\n'.join(lines).decode("utf-8")
			msg = Parser().parsestr(msg_content)
			print_info(msg)
			for part in msg.walk():
				file_name = part.get_filename()
				h = Header(file_name)
				dh = decode_header(h)
				if part.get("Content-Disposition") is None:
					continue
				file_name = dh[0][0]
				if file_name:
					num += 1
					print("=="*15)
					print("Saving %sth images..." % num)
					f_csv = open("./Data/captcha_info.csv", "a", newline="")
					w = csv.writer(f_csv)
					w.writerow([self.str, str(num) + ".jpg"])
					f_csv.close()
					file_name = "./Data/" + str(num) + ".jpg"
					file_data = part.get_payload(decode=True)
					f = open(file_name, "wb")
					f.write(file_data)
					f.close()
		server.quit()


if __name__ == '__main__':
	t = GetCaptchaData()
	t.main()