<template>
  <div class="row">
    <div v-for="card in cards" :key="card.name">
      <p>CARD: {{ card.name }}</p>
      <Card
        :alt="alt"
        :card="card.img"
        :suite="card.name"
        :height="height"
        :width="width"
      />
    </div>
  </div>
</template>

<script>
import Card from "./Card.vue";

export default {
  components: { Card },
  data: () => {
    return {
      height: "100",
      width: "100",
      alt: "asdfasdf",
      cards: [
        {
          name: "club",
          img: "2",
        },
        {
          name: "club",
          img: "3",
        },
        {
          name: "spade",
          img: "4",
        },
      ],
    };
  },
};
</script>
<style scoped>
</style>