<template>
  <img :alt="alt" :src="getImgUrl()" :height="height" :width="width" />
</template>


<script>
export default {
  props: {
    alt: String,
    card: String,
    suite: String,
    height: String,
    width: String,
  },

  methods: {
    getImgUrl() {
      var images = require.context(
        "../assets/public/static/images/cards/",
        true,
        /\.png$/
      );
      return images("./" + this.suite + "/" + this.card + ".png");
    },
  },
};
</script>
