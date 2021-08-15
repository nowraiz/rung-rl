<template>
  <!-- <img alt="Vue logo" v-bind:src="src" :height="height" :width="width"/> -->
  <!-- <HelloWorld :msg="mess" /> -->
  <!-- <Card
    :alt="alt"
    :card="card"
    :suite="suite"
    :height="height"
    :width="width"
  /> -->
  <CardStack :height="height" :width="width" position="left" />
</template>

<script>
// import HelloWorld from "./components/HelloWorld.vue";
// import image from "./assets/public/static/images/cards/club/2.png";
// import Card from "./components/Card.vue";
import CardStack from "./components/CardStack.vue";

export default {
  name: "App",
  components: {
    // HelloWorld,
    // Card,
    CardStack,
  },
  data: function () {
    return {
      src: "2",
      alt: "Some card",
      mess: "Welcome",
      height: "150",
      width: "120",
      suite: "diamond",
      card: "3",
      pos: "left",
    };
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  /* text-align: center; */
  color: #2c3e50;
  margin-top: 60px;
}
</style>
