# DOCS

## Update

### API update

```shell
# update apt docs
sphinx-apidoc -o source/_api ../ISAT/ ../ISAT/segment_any/edge_sam/* ../ISAT/segment_any/mobile_sam/* ../ISAT/segment_any/sam2/* ../ISAT/segment_any/segment_anything/* ../ISAT/segment_any/segment_anything_hq/* ../ISAT/segment_any/segment_anything_fast/* ../ISAT/segment_any/segment_anything_med2d/* ../ISAT/ui/* ../ISAT/icons_rc.py -f
```

### Internationalization

```shell
sphinx-build -b gettext ./source/ ./build/gettext/
```

```shell
sphinx-intl update -p ./build/gettext/ -l zh_CN -d source/locales/
```

## Preview

```shell
# build and preview for zh_CN
sphinx-build -b html -D language=zh_CN source ./build/html/zh_CN

# build and preview for en
sphinx-build -b html -D language=en source ./build/html/en
```
