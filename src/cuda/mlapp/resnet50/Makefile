TARGETDIR := $(BINDIR)/$(BINSUBDIR)

all: resnet

resnet: resnet.sh resnet.py
	cp resnet.sh resnet.py $(TARGETDIR)
	chmod +x $(TARGETDIR)/resnet.sh
	bash ./resnet.sh

clean:
	rm $(TARGETDIR)/resnet*