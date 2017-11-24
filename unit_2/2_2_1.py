from lxml import etree,html

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
print sys.getdefaultencoding()

path = "1.htm"
content = open(path,"rb").read()
page = html.document_fromstring(content)
print page
text = page.text_content()
print text