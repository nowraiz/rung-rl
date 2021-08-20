<template>
  <b-img :alt="alt" :src="getImgUrl(visible)" :height="height" :width="width" />
</template>


<script>
export default {
  props: {
    alt: String,
    card: String,
    suit: String,
    height: String,
    width: String,
    visible: {
      type: Boolean,
      default: false,
    },
  },

  methods: {
    getImgUrl(is_visible) {
      if (is_visible === true) {
        let images = require.context(
          "../assets/public/static/images/cards/",
          true,
          /\.png$/
        );
        return images("./" + this.suit + "/" + this.card + ".png");
      }
      // down card get
      let images = require.context(
        "../assets/public/static/images/cards/back/",
        true,
        /\.png$/
      );
      return images("./blue.png");
    },
  },
};
</script>
