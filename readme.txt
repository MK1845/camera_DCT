opencvによって取得した画像をグレースケールに変換し
離散コサイン変換をおこなうプログラム。
トラックバーの値に応じて複合に用いる係数を決定する。
トラックバーの値を10で割って整数型にキャスト変換した値が複合に用いる係数となる。
係数が小さいほどぼやけた画像となる。
係数に応じて復元した画像が表示される。

参考URL：http://bugra.github.io/work/notes/2014-07-12/discre-fourier-cosine-transform-dft-dct-image-compression/
相違点:このサイトにおいては一枚の画像をDCT変換しているが、作成したプログラムにおいては
カメラからキャプチャした画像をそのまま変換し、トラックバーの値に応じて復元したものを
opencvを用いて表示している。(画像データの変数型がImage型でなくndarray型である)

youtube_URL:https://www.youtube.com/watch?v=Jsxni3PXZQE
