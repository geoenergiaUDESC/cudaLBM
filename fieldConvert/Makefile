include ../common.mk

EXECUTABLE = fieldConvert
SOURCE = fieldConvert.cu

default: clean $(EXECUTABLE)

$(EXECUTABLE):
	$(NVCXX) $(NVCXX_FLAGS) $(SOURCE) -o $@

install: clean uninstall $(EXECUTABLE)
	cp $(EXECUTABLE) ../build/bin/
	rm -f $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)

uninstall:
	rm -f ../build/bin/$(EXECUTABLE)